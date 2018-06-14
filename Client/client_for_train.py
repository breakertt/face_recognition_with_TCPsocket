import socket
import threading
import os
import sys
from PIL import Image
import zipfile
import time

def zip_dir(dirname,zipfilename):
    filelist = []
    if os.path.isfile(dirname):
        filelist.append(dirname)
    else :
        for root, dirs, files in os.walk(dirname):
            for dir in dirs:
                filelist.append(os.path.join(root,dir))
            for name in files:
                filelist.append(os.path.join(root, name))

    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:
        arcname = tar[len(dirname):]
        #print arcname
        zf.write(tar,arcname)
    zf.close()

SIZE = 1024

if len(sys.argv) < 5:  
    print('usage: client_for_train.py TrainDataFolder_path&name ip_address port name')
    sys.exit()
    
zip_dir(sys.argv[1],(os.getcwd() + "/temp.zip"))

connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#sock.connect((sys.argv[2], int(sys.argv[3])))

connection.connect(('10.10.2.220', 22338))
connection.send('hello server'.encode())
print(connection.recv(SIZE))
connection.send("train".encode())
print(connection.recv(SIZE))
name = sys.argv[4]
connection.send(name.encode())
print(connection.recv(SIZE))
connection.send("begin to send".encode())
print("sending, please wait for a second ...")
with open((os.getcwd() + "/temp.zip"),"rb") as f:
    for data in f:
        connection.send(data)
connection.close()
time.sleep(10)
print("Train Successful\n")

'''
connection2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

connection2.connect(('10.10.2.220', 22338))
print("sent, waiting for respond...")
connection2.send('hello server'.encode())
print(connection2.recv(SIZE))
connection2.send("server to client".encode())
while True:
    data = connection2.recv(SIZE)
    if not data:
        print("reach the end of file")
        break
    elif data == "begin to respond".encode():
        print("create file")
        with open("%s_output.jpg" % portion[0], "wb") as f:
            data = None
            pass
    else:
        with open("%s_output.jpg" % portion[0], "ab") as f:
            f.write(data)
            data = None
print("get respond successfully")
connection2.close()
img = Image.open("%s_output.jpg" % portion[0])
img.show()'''
print("connection closed")