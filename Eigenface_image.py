import pickle
import cv2
import numpy as np

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier('D:\\face-recognition-eigenfaces\\haarcascade_frontalface_default.xml')

    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces)==0):
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y+w,x:x+h], faces[0]

def draw_rect(image,rect):
    x,y,w,h = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
def text(image,text,rect):
    x,y,w,h = rect
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)

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

#resizing training image. Because Eigen face can not train with various sizes of photo
faces_resize = [cv2.resize(face,(500,500)) for face in faces]

#Creating Eigen Face Recognizer
eigen_recognizer = cv2.face.EigenFaceRecognizer_create()

#Traing Eignen Face recognizer
eigen_recognizer.train(faces_resize,np.array(labels_int))

image = cv2.imread(test_path)
face, rect = detect_face(image)

#Resize testing image. Because testing image should be same size as traing images size
resized_test_face = cv2.resize(face,(500,500))
l,c = eigen_recognizer.predict(resized_test_face)
print("label",l)
print("confidence",c)

draw_rect(image,rect)
text(image,anti_dic[l],rect)

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()  