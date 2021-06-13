from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

dataset="AI(SmartAttendenceSystem)/dataset"

embeddingFile="AI(SmartAttendenceSystem)/output/embeddings.pickle"#initial name for embedding file
embeddingModel="AI(SmartAttendenceSystem)/openface_nn4.small2.v1.t7"#initializing model for embedding pyt

#initialization of caffe model for face detection
prototxt="AI(SmartAttendenceSystem)/model/MobileNetSSD_deploy.prototxt"
model="AI(SmartAttendenceSystem)/model/res10_300x300_ssd_iter_140000.caffemodel"

#loading caffe model for face detection
#detecting face from Image via caffe deep learning
detector=cv2.dnn.readNetFromCaffe(prototxt,model)

#loading pytorch model file for extract facial embeddings
#extracting facial embeddings via deep learning feature extracting
embedder=cv2.dnn.readNetFromCaffe(embeddingModel)

#getting image paths
imagePaths=cv2.dnn.readNetFromTorch(embeddingModel)

#initialization
knownEmbeddings=[]
knownNames=[]
total=0
conf=0.5

#we start to read images one by one to apply face detection and embedding
for (i,imagePath) in enumerate(imagePaths):
    print("Processing image {}/{}".format(i+1,len(imagePaths)))
    name=imagePath.split(os.path.sep)[-2]
    image=imutils.resize(image,,width=600)
    (h,w)=image.shape[:2]
    #converting image to blob for dnn face detection
    imageBlob=cv2.dnn.blobFromImages(cv2.resize(image,(300,300),1.0,(300,300),(104.0,177.0,123.0)))
    
    #setting input blob image
    detector.setInput(imageBlob)
    #prediction the face
    detections=detector.forward()
    
    if len(detections) > 0:
        i=np.argmax(detections[0,0,:,2])
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
            vec=embedder.forward()#vec-vector
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total +=1
            
print("Embeddings:[0]".format(total))
data={"embeddings":knownEmbeddings,"names":knownNames}
f=open(embeddingFile,"wb")#wb-write binary
f.write(pickle.dumps(data))
f.close()
print("Process Completed")
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    