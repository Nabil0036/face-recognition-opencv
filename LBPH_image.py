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

dic = {"Nabil": 0,"Tom Hanks": 1}
anti_dic = {0:'Nabil', 1:'Tom Hanks'}
labels_int = [dic[label] for label in labels]

#Creating LBPH(Local Binary Pattern Histogram) Face Recognizer
lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Train LBPH recognizer 
lbph_recognizer.train(faces,np.array(labels_int))

image = cv2.imread(test_path)

ess = Main()

face, rect = ess.detect_face(image)

#predict new test face
l,c = lbph_recognizer.predict(face)
print("label",l)
print("confidence",c)

ess.draw_rect(image,rect)
ess.text(image,anti_dic[l],rect)

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()  