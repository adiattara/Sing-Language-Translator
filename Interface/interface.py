import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import pandas as pd

# Charger le modèle sauvegardé
loaded_model = tf.saved_model.load('../model_asl')

infer = loaded_model.signatures["serving_default"]
# Fonction pour effectuer l'inférence sur les données d'entrée
def predict_sequence(sequence):
    # Assurez-vous que les données de la séquence ont la forme correcte
    input_data = np.array(sequence)  # Assurez-vous que sequence a la forme (30, 1662)

    # Redimensionner les données de la séquence pour avoir la forme (64, 3258)
    input_data = np.resize(input_data, (64, 3258))

    input_data = tf.constant(input_data, dtype=tf.float32)

    # Effectuer l'inférence
    output = infer(inputs=input_data)

    # Extraire les prédictions
    predictions = output["output_0"].numpy()
    predicted_class = np.argmax(predictions, axis=-1)
    return predicted_class


# Charger le signe-to-prediction map
def load_json_file(json_path):
    with open(json_path, 'r') as f:
        sign_map = json.load(f)
    return sign_map

class CFG:
    data_dir = "./"
    sequence_length = 12
    rows_per_frame = 543

sign_map = load_json_file(CFG.data_dir + 'sign_to_prediction_index_map.json')
s2p_map = {k.lower(): v for k, v in sign_map.items()}
p2s_map = {v: k for k, v in sign_map.items()}

train_data = pd.read_csv(CFG.data_dir + 'train.csv')
actions = train_data["sign"].tolist()
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Définir les fonctions utilitaires
def draw_landmarks(image, results):
    if results.face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style())
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style())

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Conversion couleur BGR vers RGB
    image.flags.writeable = False  # L'image n'est plus modifiable
    results = model.process(image)  # Faire une prédiction
    image.flags.writeable = True  # L'image est maintenant modifiable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Conversion couleur RGB vers BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    predicted_action = actions[res[0]]  # Récupère l'action prédite
    cv2.putText(output_frame, predicted_action, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Inférence en temps réel
sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture(0)
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            predicted_class = predict_sequence(sequence)
            action = actions[predicted_class[0]]
            print(action)

            if len(sentence) > 0:
                if action != sentence[-1]:
                    sentence.append(action)
            else:
                sentence.append(action)

            if len(sentence) > 5:
                sentence = sentence[-5:]

            image = prob_viz(predicted_class, actions, image)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()