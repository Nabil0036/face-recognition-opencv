import cv2
import numpy as np
import os
from itertools import chain
import pickle
from essentials import Main
data_dir = 'D:\\face-recognition-eigenfaces\\dataset'

#name of the training faces
names=[name for name in os.listdir(data_dir)]

#take all the images and their label as a tuple in a 2d list
image_paths =[[(data_dir+'\\'+name+'\\'+path,name) for path in os.listdir(data_dir+'\\'+name)] for name in names]

#flatten the 2d list
image_paths = list(chain.from_iterable(image_paths))
# print(image_paths)

ess = Main()

faces,labels = ess.prepare_data(image_paths)
print(faces[0])
print(labels[0])

f = open("faces.pickle","wb")
pickle.dump(faces,f)
f.close()

l = open("labels.pickle","wb")
pickle.dump(labels, l)
l.close()