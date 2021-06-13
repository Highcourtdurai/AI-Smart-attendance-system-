import imutils
import numpy as np
import imutils
import pickle
import time
import cv2

embeddingModel="AI(SmartAttendenceSystem)/openface_nn4.small2.v1.t7"

embeddingFile="AI(SmartAttendenceSystem)/output/embeddings.pickle"
recognizerFile="AI(SmartAttendenceSystem)/output/recognizer.pickle"
labelEncFile="AI(SmartAttendenceSystem)/output/le.pickle"
conf=0.5

print("Loading face detector...")
prototxt="AI(SmartAttendenceSystem)/model/MobileNetSSD_deploy.prototxt"
model="AI(SmartAttendenceSystem)/model/res10_300x300_ssd_iter_140000.caffemodel"
detector=cv2.dnn.readNetFromCaffe(prototxt,model)

print("Loading face recognizer...")
embedder=cv2.dnn.readNetFromTorch(embeddingModel)

recognizer=pickle.loads(open(recognizerFile,"rb").read())
le=pickle.loads(open(labelEncFile,"rb").read())#le-label encoder

box=[]
print("Starting video stream...")
cam=cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _,frame=cam.read()
    frame=imutils.resize(frame,width=400)
    (h,w)=image.shape[:2]
    imageBlob=cv2.dnn.blobFromImages(cv2.resize(image,(300,300),1.0,(300,300),(104.0,177.0,123.0)))
    
    detector.setInput(imageBlob)
    detections=detector.forward()
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        
         if confidence > conf:
            #ROI range of interest
            box=detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")
            face=image[startY:endY,startX:endX]
            (fH,fW)=face.shape[:2]
            
            if fW < 20 or fH <20:
                continue
            #image to blob for face
            faceBlob=cv2.dnn.blobFromImage(face,1.0/255,(96,96),(0,0,0),swapRB=True,)
            #facial features embedder input image
            embedder.setInput(faceBlob)
            vec=embedder.forward()
            
            preds=recognizer.predict_proba(vec)[0]
            j=np.argmax(preds)
            proba=preds()
            name=le.classes_[j]
            text="{} : {:.2f}%".format(name,proba 8 100)
            y=startY-10 if startY -10 >10 else startY +10
            cv2.rectangle(frame,(startX,startY),(endX,endY),(0,0,255),2)
            cv2.putText(frame,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255)
    cv2.imshow("Frame",frame)
    key=cv2.waitkey(1) & 0xFF
    if key==ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    
    
    
    
