from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

#SVC-support vector classifier

#initializing of embedding & recognizer
embeddingFile="AI(SmartAttendenceSystem)/output/embeddings.pickle"
#New & Empty at initial
recognizerFile="AI(SmartAttendenceSystem)/output/recognizer.pickle"
labelEncFile="AI(SmartAttendenceSystem)/output/le.pickle"

print("Loading face embeddies...")
data=pickle.loads(open(embeddingFile,"rb").read())


print("Encoding labels....")
labelEnc=LabelEncoder()
labels=labelEnc.fit_transform(data["names"])

print("Training model...")
recognizer=SCV(C=1.0,kernel="linear",probability=True)
recognizer=fit(data["embeddings"],labels)

f=open(recognizerFile,"wb")
f.write(pickle.dumps(recognizer))
f.close()

f=open(labelEncFile,"wb")
f.write(pickle.dumps(labelEnc))
f.close()


