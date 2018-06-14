import socket
import threading
import os
import sys
from PIL import Image


SIZE = 1024

if len(sys.argv) < 4:  
    print('usage: client.py img_path&name ip_address port')
    sys.exit()
pic_filename = sys.argv[1]

#pic_filename = './test.jpg'
img = Image.open(pic_filename)
portion = os.path.splitext(pic_filename)
img.save("./temp.jpg", quality = 60)
connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#sock.connect((sys.argv[2], int(sys.argv[3])))

connection.connect(('10.10.2.220', 22338))
connection.send('hello server'.encode())
print(connection.recv(SIZE))
connection.send("client to server".encode())
print(connection.recv(SIZE))
connection.send("begin to send".encode())
print("sending, please wait for a second ...")
with open("./temp.jpg","rb") as f:
    for data in f:
        connection.send(data)
print("close connection for receiving result")
connection.close()

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
img.show()
print("connection closed")