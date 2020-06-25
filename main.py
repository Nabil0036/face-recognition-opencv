import cv2
import numpy as np
import os
from itertools import chain
import pickle

data_dir = 'D:\\face-recognition-eigenfaces\\dataset'

#name of the training faces
names=[name for name in os.listdir(data_dir)]

#take all the images and their label as a tuple in a 2d list
image_paths =[[(data_dir+'\\'+name+'\\'+path,name) for path in os.listdir(data_dir+'\\'+name)] for name in names]

#flatten the 2d list
image_paths = list(chain.from_iterable(image_paths))
# print(image_paths)

def prepare_data(image_paths):
    faces = []
    labels =[]
    i=0
    for image_path in image_paths:
        image = image_path[0]
        label = image_path[1]
        
        img = cv2.imread(image)

        face, rect = detect_face(img)
        print(".")
        if face is not None:
            faces.append(face)
            labels.append(label)
        else:
            i+=1
            print(i)
    
    return faces, labels
        


#prepare_data(image_paths)
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier('D:\\face-recognition-eigenfaces\\haarcascade_frontalface_default.xml')

    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces)==0):
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y+w,x:x+h], faces[0]


faces,labels = prepare_data(image_paths)
print(faces[0])
print(labels[0])

f = open("faces.pickle","wb")
pickle.dump(faces,f)
f.close()

l = open("labels.pickle","wb")
pickle.dump(labels, l)
l.close()