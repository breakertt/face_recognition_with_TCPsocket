#!/usr/bin/python  
#coding=utf-8  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
import csv
import socket
import threading
import zipfile
import shutil

def restart_program():
  python = sys.executable
  os.execl(python, python, * sys.argv)

def un_zip(file_name,dir_name):
    #unzip zip file
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(file_name + "_files"):
        pass
    else:
        os.mkdir(file_name + "_files")
    for names in zip_file.namelist():
        zip_file.extract(names,file_name + "_files/")
    os.rename((file_name + "_files"),"./mydataset/temp/origin/%s" % dir_name)
    zip_file.close()

def checkFile():
    list = os.listdir('./mydataset/test')
    for iterm in list:
        if iterm == 'test.jpg':
            os.remove(iterm)
            print('remove')
            break
        else:
            pass
        
def tcplink(connection, addr):
    global update
    print("Accept new connection from %s:%s" % addr)
    connection.send("Welcome from server!")
    print(connection.recv(SIZE))
    while True:
        data = connection.recv(SIZE)
        if data=="client to server":  #receive
            connection.send("begin to receive")
            print("receiving, please wait")
            while True:
                data = connection.recv(SIZE)
                if not data :
                    print("reach the end of file")
                    break
                elif data == 'begin to send':
                    print("create file")
                    #checkFile()
                    with open("./mydataset/test/test.jpg", "wb") as f:
                        data = None
                        pass
                else:
                    with open("./mydataset/test/test.jpg", "ab") as f:
                        f.write(data)
                        data = None
            connection.close()
        elif data=="server to client":   #send
            print("received, start recognize!")
            #os.system("python pic_face_dete.py")
            connection.send('begin to respond')
            print('sending, please wait for a second ...')
            with open('./mydataset/test/test_output.jpg', 'rb') as f:
                for data in f:
                    connection.send(data)
            print("responded! Connection from %s:%s closed." % addr)
            connection.close()
        elif data=="train":
            os.makedirs("./mydataset/temp")
            os.makedirs("./mydataset/temp/origin")
            os.makedirs("./mydataset/temp/train")
            connection.send("begin to receive name")
            name = connection.recv(SIZE)
            connection.send("begin to receive images")
            print("receiving, please wait")
            while True:
                data = connection.recv(SIZE)
                if not data :
                    print("reach the end of file")
                    break
                elif data == 'begin to send':
                    print("create file")
                    #checkFile()
                    with open("./mydataset/temp/origin/temp.zip", "wb") as f:
                        data = None
                        pass
                else:
                    with open("./mydataset/temp/origin/temp.zip", "ab") as f:
                        f.write(data)
                        data = None
            connection.close()
            with open("./mydataset/count.txt", 'r') as f:
                count = int(f.readline()) + 1
            with open("./mydataset/count.txt", 'w') as f:
                f.write(str(count))
            with open("./models/human_name.csv", 'a+') as f:
                f.write(str(count).zfill(6) + "," + name + "\n")
            os.system("python trainHelper.py")
            #os.execl('restartserver.sh', '')
            update = 1
        break

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('10.10.2.220', 22338)
#server_socket.settimeout(1)
server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
server_socket.bind(server_address) 
server_socket.listen(1)
print("waiting for connection..")
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './models/')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        SIZE = 1024
        update = 0  #update .pkl
        with open("./models/human_name.csv", "r") as f:
            human_name_reader = csv.reader(f)
            HumanNames = [row[1] for row in human_name_reader]
            del HumanNames[0]            
        #HumanNames = [line.rstrip('\n') for line in open("./models/human_name.txt")]
        
        #test_csv = open("./mydataset/test/test.csv","w")
        #test_csv.close()
        
        print('Loading feature extraction model')
        modeldir = './models/20180402-114759.pb'
        facenet.load_model(modeldir)
        classifier_filename = './models/my_classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        c = 0

        while True:
            list = os.listdir('./mydataset/test')
            for iterm in list:
                if iterm == 'test.jpg':
                    print('Start Recognition!')
                    prevTime = 0
                    i = 0
                    while i==0:
                        #ret, frame = video_capture.read()

                        frame = cv2.imread("./mydataset/test/test.jpg")
                    
                    #curTime = time.time()    # calc fps
                        timeF = 3

                        if (c % timeF == 0):
                            find_results = []

                            if frame.ndim == 2:
                                frame = facenet.to_rgb(frame)
                            frame = frame[:, :, 0:3]
                            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                            nrof_faces = bounding_boxes.shape[0]
                            print('Detected_FaceNum: %d' % nrof_faces)

                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                img_size = np.asarray(frame.shape)[0:2]

                                cropped = []
                                scaled = []
                                scaled_reshape = []
                                bb = np.zeros((nrof_faces,4), dtype=np.int32)

                                for i in range(nrof_faces):
                                    emb_array = np.zeros((1, embedding_size))

                                    bb[i][0] = det[i][0]
                                    bb[i][1] = det[i][1]
                                    bb[i][2] = det[i][2]
                                    bb[i][3] = det[i][3]

                                    # inner exception
                                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                        print('face is inner of range!')
                                        continue

                                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                                    cropped[0] = facenet.flip(cropped[0], False)
                                    scaled.append(misc.imresize(cropped[0], (image_size, image_size), interp='bilinear'))
                                    scaled[0] = cv2.resize(scaled[0], (input_image_size,input_image_size),
                                                        interpolation=cv2.INTER_CUBIC)
                                    scaled[0] = facenet.prewhiten(scaled[0])
                                    scaled_reshape.append(scaled[0].reshape(-1,input_image_size,input_image_size,3))
                                    feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                    predictions = model.predict_proba(emb_array)
                                    best_class_indices = np.argmax(predictions, axis=1)
                                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                    #plot result idx under box
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    print('result: ', best_class_indices)
                                    for H_i in HumanNames:
                                        if HumanNames[best_class_indices[0]] == H_i:
                                            result_names = HumanNames[best_class_indices[0]]
                                            cv2.putText(frame, result_names, (text_x, text_y),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),5)
                            else:
                                print('Unable to align')


                        cv2.imwrite("./mydataset/test/test_output.jpg",frame)  
                        i = 1
            connection, addr = server_socket.accept()
            print(update)
            if update == 1:
                with open("./models/human_name.csv", "r") as f:
                    human_name_reader = csv.reader(f)
                    HumanNames = [row[1] for row in human_name_reader]
                    del HumanNames[0]            
                classifier_filename = './models/my_classifier.pkl'
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                    print('load classifier file-> %s' % classifier_filename_exp)
                update = 0
            t = threading.Thread(target = tcplink, args = (connection, addr))
            t.setDaemon(True)
            t.start()
            t.join()


                    


        #video_capture.release()
        # #video writer
        # out.release()
        cv2.destroyAllWindows()
