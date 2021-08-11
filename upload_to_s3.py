import os
import json
import glob
import boto3
from botocore.exceptions import ClientError



mac_id = 'id_f4:96:34:02:75:dd'
s3_segmentation_dir = 'nano_segmentation'
bucket_name = 'care-plus'

test_path = os.path.join(bucket_name, mac_id)

### this section is used to upload file to s3

destination_path_s = os.path.join(mac_id, s3_segmentation_dir, 's.png')
destination_path_f = os.path.join(mac_id, s3_segmentation_dir, 'f.png')
destination_path_json = os.path.join(mac_id, s3_segmentation_dir, 'f.json')

s3 = boto3.client('s3')

with open('s.png', 'rb') as s:
    s3.upload_fileobj(s, bucket_name, destination_path_s)
with open('f.png', 'rb') as f:
    s3.upload_fileobj(f, bucket_name, destination_path_f)
with open('f.json', 'rb') as f_json:
    s3.upload_fileobj(f_json, bucket_name, destination_path_json)


### s3 test
"""
client = boto3.client('s3')
response = client.list_objects(
    Bucket=bucket_name,
    Delimiter=mac_id,
    MaxKeys=1,
)
print(response)
print(len(response))

s3 = boto3.resource('s3')
my_bucket = s3.Bucket(bucket_name)
#print(len(my_bucket.objects.all()))


#sum = 0
#for my_bucket_object in my_bucket.objects.all():
#    sum += 1
#    print(sum)    

for key in client.list_objects(Bucket=bucket_name)['Contents']:
    print(key['Key'])
"""