import pickle
import cv2
import numpy as np

class Main():
    def __init__(self):
        pass


    def detect_face(self,image):
        self.image = image
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier('D:\\face-recognition-eigenfaces\\haarcascade_frontalface_default.xml')

        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if (len(faces)==0):
            return None, None

        (x, y, w, h) = faces[0]

        return gray[y:y+w,x:x+h], faces[0]

    def draw_rect(self,image,rect):
        self.rect = rect
        self.image = image 
        x,y,w,h = self.rect
        cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    def text(self,image,text,rect):
        self.image = image 
        self.text = text
        self.rect = rect
        x,y,w,h = self.rect
        cv2.putText(self.image, self.text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)


    def prepare_data(self,image_paths):
        self.image_paths = image_paths
        faces = []
        labels =[]
        i=0
        for image_path in self.image_paths:
            image = image_path[0]
            label = image_path[1]
            
            img = cv2.imread(image)

            face, rect = self.detect_face(img)
            print(".")
            if face is not None:
                faces.append(face)
                labels.append(label)
            else:
                i+=1
                print(i)
        
        return faces, labels