import paramiko
from paramiko import SSHClient
from scp import SCPClient
import os.path, os
from ftplib import FTP, error_perm



### scp, ssh 
"""
ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect(hostname='192.168.0.92', 
            port = '5000',
            username='kenchang',
            password='meteorx900824')
            #pkey='load_key_if_relevant')


# SCPCLient takes a paramiko transport as its only argument
scp = SCPClient(ssh.get_transport())

#scp.put('file_path_on_local_machine', 'file_path_on_remote_machine')
#scp.get('file_path_on_remote_machine', 'file_path_on_local_machine')

scp.close()
"""


### https://stackoverflow.com/questions/32481640/how-do-i-upload-full-directory-on-ftp-in-python
host = '192.168.0.92'
port = 5000

ftp = FTP()
ftp.connect(host)
ftp.login('kenchang', 'meteorx900824')
#filenameCV = "directorypath"
file_path = '/home/harvey/activesample/care-plus-model-pipeline-activesample_dev/up_test_as_user/'
remote_path = '/data/user_data/user01'

"""
for name in os.listdir(file_path):
    print('name', name)
    localpath = os.path.join(file_path, name)
    print(os.path.isdir(file_path))
    print(os.path.isdir(localpath))
"""

def placeFiles(ftp, path, remote_path):
    for name in os.listdir(path):
        localpath = os.path.join(path, name)
        if os.path.isfile(localpath):
            print("STOR", name, localpath)
            ftp.storbinary('STOR ' + remote_path + '/' + name, open(localpath,'rb'))
        elif os.path.isdir(localpath):
            print("MKD", name)
            ftp.mkd(remote_path + '/' + name)
            try:
                ftp.mkd(name)
                print('success mkd')

            # ignore "directory already exists"
            except error_perm as e:
                if not e.args[0].startswith('550'): 
                    raise

            print("CWD", name)
            ftp.cwd(name)
            placeFiles(ftp, localpath, remote_path)           
            print("CWD", "..")
            ftp.cwd("..")

placeFiles(ftp, file_path, remote_path)

ftp.quit()
