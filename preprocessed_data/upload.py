import os
import boto3
from mimetypes import MimeTypes

def upload_files(directory, bucket_name):
    s3_client = boto3.client('s3')
    mime = MimeTypes()

    # Iterate through the files in the directory
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(full_path, directory)
            content_type, _ = mime.guess_type(full_path)

            try:
                print(f'Uploading {full_path} to s3://{bucket_name}/{relative_path}')
                s3_client.upload_file(
                    Filename=full_path,
                    Bucket=bucket_name,
                    Key=relative_path,
                    ExtraArgs={'ContentType': content_type or 'binary/octet-stream'}
                )
            except Exception as e:
                print(f'Failed to upload {full_path}: {e}')

if __name__ == "__main__":
    local_directory = "./"  # Replace with your local directory
    s3_bucket_name = "iabd-pa-project"         # Replace with your S3 bucket name
    upload_files(local_directory, s3_bucket_name)
