import pickle
import cv2
import numpy as np
from essentials import Main

test_path = "D:\\face-recognition-eigenfaces\\test\\11.jpg"

#loads the faces from the pickle file 
f = open("faces.pickle","rb")
faces = pickle.load(f)
f.close()

#loads the labels from the pickle file
l = open("labels.pickle","rb")
labels = pickle.load(l)
l.close()

#print(len(faces),len(labels))
dic = {"Nabil": 0,"Tom Hanks": 1}
anti_dic = {0:'Nabil', 1:'Tom Hanks'}

#convert string to integer
labels_int = [dic[label] for label in labels]

#resizing training image. Because Fisher face can not train with various sizes of photo
faces_resize = [cv2.resize(face,(500,500)) for face in faces]

#Creating Fisher Face Recognizer
fisherface_recognizer = cv2.face.FisherFaceRecognizer_create()

#Traing Fisher Face recognizer
fisherface_recognizer.train(faces_resize,np.array(labels_int))

ess = Main()

image = cv2.imread(test_path)
face, rect = ess.detect_face(image)

#Resize testing image. Because testing image should be same size as traing images size
resized_test_face = cv2.resize(face,(500,500))
l,c = fisherface_recognizer.predict(resized_test_face)
print("label",l)
print("confidence",c)

ess.draw_rect(image,rect)
ess.text(image,anti_dic[l],rect)

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()  