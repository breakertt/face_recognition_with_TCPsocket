import socket
import threading
import os
import zipfile
import shutil

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

with open("./mydataset/count.txt", 'r') as f:
    count = int(f.readline())
newtrain_dir = "Human_%s" % str(count).zfill(6)
un_zip("./mydataset/temp/origin/temp.zip",newtrain_dir)
os.system("python Make_aligndata_git.py %s" % newtrain_dir)
shutil.move("./mydataset/temp/train/%s" % newtrain_dir,"./mydataset/train/%s" % newtrain_dir)
shutil.move("./mydataset/temp/origin/%s" % newtrain_dir,"./mydataset/origin/%s" % newtrain_dir)
#os.makedirs("./mydataset/origin/%s" % newtrain_dir) 
os.system("python Make_classifier_git.py")
shutil.rmtree("./mydataset/temp")
#os.execl('restartserver.sh', '')