import os
import boto3
import csv
#import pandas as pd
from botocore.exceptions import ClientError
from smb.SMBConnection import SMBConnection


### NAS
"""
host = "192.168.0.92"  # ip
username = "kenchang"
password = "meteorx900824"
wanted_user = 'user04'
"""

### S3
"""
mac_id = 'id_f8:59:71:98:90:18'
bucket_name = 'care-plus'
prefix = mac_id+"/"+"nano_prediction"+"/"
"""

### NAS_list
def get_NAS_list(wanted_user, host, username, password):
    conn = SMBConnection(username, password, "", "", use_ntlm_v2 = True)
    result = conn.connect(host, 445)
    if result:
        file_path = conn.listPath('data', '/user_data/' + wanted_user)
        nas_list = []
        for x in file_path:
            #print(x.filename)
            x_split = x.filename.split()
            nas_list.append(x_split)

        nas_list = sorted(nas_list)
    else:
        print('Not connect')
    
    return nas_list

### AWS_S3
def get_s3_list(bucket_name, prefix):
    client = boto3.client('s3')
    result = client.list_objects(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    aws_list = []
    for o in result.get('CommonPrefixes'):
        aws_list.append(o.get('Prefix')[37:47])
        
    return aws_list

def get_user_id_list(id_csv):
    """ input: a csv file
        return: a user_list[] 
    """
    with open(id_csv, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        user_data = []
        for row in rows:
            if len(row[1]) > 0:
                user_data.append(row)

    del user_data[0]
    
    return user_data



if __name__ == '__main__':
    user_list = get_user_id_list('./id.csv')
    wanted_user = 8
    print(user_list[wanted_user][1])

    ### S3_parameters
    mac_id = user_list[wanted_user][1]
    bucket_name = 'care-plus'
    prefix = mac_id+"/"+"nano_prediction"+"/"

    ### NAS parameters
    host = "192.168.0.92"  # ip
    username = "kenchang"
    password = "meteorx900824"
    #wanted_user = 'user04'

    print('user0' + str(wanted_user))
    NAS_date = get_NAS_list('user0' + str(wanted_user), host, username, password)
    s3_date = get_s3_list(bucket_name, prefix)
    #print(s3_date)
    #print(NAS_date)
    
    
    NAS_date_list = []



    for i in enumerate(NAS_date):
        if i[0] % 2 == 0:
            NAS_date_list.append(i[1])

    del NAS_date_list[0]
        
    s3_date_list = list(s3_date)
    NAS_index = []

    for x in range(len(NAS_date_list)):
        NAS_index.append(NAS_date_list[x][0])

    compared_result = []

    for data in s3_date_list:
        if data not in NAS_index:
            compared_result.append(data)
    
    print('data on s3 not in NAS', compared_result)
