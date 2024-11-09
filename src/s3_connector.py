import boto3
from botocore.exceptions import ClientError
import logging

class S3Connector:
    def __init__(self, bucket_name="cleanbagel-datasets"):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.logger = logging.getLogger(__name__)

    def list_datasets(self, prefix=""):
        """List all datasets in the specified S3 bucket/prefix"""
        print('list_datasets')
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            print('success')
            # Filter out .git directories and ensure files have valid extensions
            return [obj['Key'] for obj in response.get('Contents', [])
                    if not '/.git/' in obj['Key'] 
                    and obj['Key'].split('.')[-1] in ['csv', 'json', 'txt', 'jsonl']]  # add more extensions as needed
        except ClientError as e:
            self.logger.error(f"Error listing datasets: {e}")
            raise

    def download_dataset(self, key, local_path):
        """Download a specific dataset from S3"""
        try:
            self.s3_client.download_file(self.bucket_name, key, local_path)
            return local_path
        except ClientError as e:
            self.logger.error(f"Error downloading dataset {key}: {e}")
            raise 