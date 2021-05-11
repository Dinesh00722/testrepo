# testrepo
from keras.models import load_model
import cv2
import numpy as np
import os

#load best performing CNN model
model = load_model(r'C:\Users\Admin\model-016.model')

#Frontal Face detection classifier
face_clsfr=cv2.CascadeClassifier(r'C:\Users\Admin\python\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

#capture is webcam
cap=cv2.VideoCapture(0)

#set labels and colour for rectangle
labels_dict={0:'without_mask',1:'with_mask'}
color_dict={0:(255,0,0),1:(0,255,0)}

while(True):
    #read video stream from web cam
    ret,img=cap.read()
    
    #convert into grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #run facial detections models and return X/y co-ords and weight/height of rectangle
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:
        
        #performing pre-processing on face image(resize and normalise pixel range)
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        
        # perform prediction
        result=model.predict(reshaped)
        #label is one of higher probability
        label=np.argmax(result,axis=1)[0]
        #draw rectangle around our image,choose colour,place label
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    # Stop if escape key is pressed  
    key=cv2.waitKey(1) & 0xff
    
    if(key==27):
        break
        
cap.release()
cv2.destroyAllWindows()
