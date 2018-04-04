import numpy as np
from math import sqrt
from sklearn.externals import joblib  #for opening the trained model
import mfcc                           #python file containing code for mfcc
import scipy.io.wavfile as wav        #for reading wav file
import sounddevice as sd              #for recording and playing
import pyttsx3                        #for voice
######################################################
#This set the voice which will be used later
engine=pyttsx3.init()
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
######################################################
filename=[]                  #list that will contain the inforamtion regarding the selected voice sample
distance=[]
prediction=[]
templist1=[]
templist2=[]
sequence=['First','Second']
#sample_rate,signal = wav.read('an4.wav')
#filename.append([sample_rate,signal])
#sample_rate,signal = wav.read('rk4.wav')
#filename.append([sample_rate,signal])
speakers=['rk2.wav','an2.wav']
######################################################
'''
FUNCTION NAME:play_voice_sample_of_each_speaker
INPUT:NONE
OUTPUT:NONE
LOGIC:It simply read the wave file stored in speakers list and then play that file.
'''  
def play_voice_sample_of_each_speaker():                       #play_voice_sample_of_each_speaker()
    for speaker in range(len(speakers)):
        engine.say('voice sample of'+sequence[speaker]+'speaker')
        engine.setProperty('rate',120)
        engine.setProperty('volume',0.9)
        engine.runAndWait()
        sample_rate,signal = wav.read(speakers[speaker])
        sd.play(signal,sample_rate)
        sd.wait()
        if speaker==1:
            engine.say('now select voice sample from lvq folder by clicking on testing button and see whether lvq system is working fine or not')
            engine.setProperty('rate',120)
            engine.setProperty('volume',0.9)
            engine.runAndWait()
#####################################################

#####################################################
'''
FUNCTION NAME:euclidean_distance
INPUT:Two row between which euclidean distance is to be measured
OUTPUT:euclidean distance 
LOGIC:It simply find the euclidean distance between two rows by using euclidean formula
'''  
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(12):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
#####################################################

#####################################################
'''
FUNCTION NAME:get_best_matching_unit1
INPUT:codebook vector and test row
OUTPUT:row of codebook vector whose euclidean distance from test row is minimum.
'''  
def get_best_matching_unit1(codebooks, test_row):
    distances = list()
    distance.clear()
    for codebook in codebooks:
        dist = euclidean_distance(codebook, test_row)
        distances.append((codebook, dist))
        distance.append(dist)
    distances.sort(key=lambda tup: tup[1])
    return distances[0]
#####################################################

#####################################################
'''
FUNCTION NAME:get_best_matching_unit1
INPUT:codebook vector and test row
OUTPUT:row of codebook vector whose euclidean distance from test row is minimum.
''' 
def predict(codebooks, test_row):
    templist1.clear()
    templist2.clear()
    templist1.append(get_best_matching_unit1(codebooks, test_row))
    templist2.append(templist1[0][0][-1])
    templist2.append(templist1[0][1])
    return templist2
#####################################################

#####################################################
'''
FUNCTION NAME:find
INPUT:NONE
OUTPUT:list that will contain information regarding the identity of the test voice sample.
''' 
def find():
    result=[]                #temp list that will contain information regarding the identity of the test voice sample.
    for l in range(3):       #this loop will run for n no. of times, where n depends on no. of speakers.
        prediction.clear()
        codebook=np.load('vc6-15587.npy') #loading the trained model
        codebook=codebook.tolist()        #converting array into list
        sc=joblib.load('fi.sav')          #loading the trained model
        feat=[]                           #temp list that will hold mfcc feature
        mfcc_feat=mfcc.findmfcc(filename[0])
        temp_list=[]    
        mfcc_feat=sc.fit_transform(mfcc_feat)     #scaling the mfcc features      
        mfcc_feat=mfcc_feat.tolist()              
        count=0                           #temp variable used for counting     
        mfc=mfcc_feat[0:194]
        feat.append(mfc)
        for row in range(len(mfc)):
            g=predict(codebook,feat[0][row])
            temp_list.append(g[0])
        for j in range(len(temp_list)):
            if temp_list[j]==l:
                count=count+1           
        prediction.append(count/len(feat[0]))
        temp_list.clear()
        count=0
        mfc=mfcc_feat[len(mfcc_feat)-172:]
        for row in range(len(mfc)):
            temp_list.append((predict(codebook,mfc[row]))[0])        
        for j in range(len(temp_list)):
            if temp_list[j]==l:
                count=count+1 
        prediction.append(count/len(mfc))
        result.append(max(prediction))   
    return result
####################################################