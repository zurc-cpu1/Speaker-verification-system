import numpy as np
from scipy.cluster.vq import vq          #for applying vq
from sklearn.cluster import KMeans       #for applying kmeans
from sklearn.preprocessing import StandardScaler          #for scaling the mfcc features
import mfcc                              #python file containing code for mfcc
from math import sqrt
import scipy.io.wavfile as wav           #for reading wave file
sc=StandardScaler()
file=[]                                  #list that will hold the voice information of each voice samples
#################################################
'''
FUNCTION NAME:train
INPUT:mfcc feature of each voice sample
OUTPUT:list containing the informtion regarding clusters center and numpy array  
       containing distance of each frame from its nearest cluster and the cluster number
LOGIC:Each mfcc_features is divided into n clusters and then the distance of each frame from its nearest cluster 
      is calculated.
'''    
def train(data):
    trained_data=[]          #temperory list that will hold the trained features
    estimator=KMeans(n_clusters=128)
    estimator.fit(data)
    t=estimator.cluster_centers_         #t contains the information of each cluster center
    trained_data=[t,vq(data,t)]          
   # trained_data.append(vq(data,t))
    return trained_data
#################################################
    
#################################################
'''
FUNCTION NAME:calculate
INPUT:multiple list containing the idea regarding the identity of each speaker.
OUTPUT:list containing the idea regarding the identity of test voice sample.
LOGIC:multiple list is reduced into single list by summing up the element of each column of every row
''' 
def calculate(test_result):
    result=[]    #temp list that will contain information regarding the identity of the test voice sample
    for speakers in range(len(test_result[0])):
        temp_variable=0.0          #temp variable used to store the sum
        for count in range(len(test_result)):
            temp_variable +=test_result[count][speakers]
        result.append(temp_variable)    
    for speakers in range(len(result)):
        result[speakers]=result[speakers]/len(test_result)
    return result
###############################################
    
###############################################
'''
FUNCTION NAME:test_for_recognition
INPUT:NONE
OUTPUT:List containing information regarding the validity of the test voice sample.
LOGIC:trained_mfcc_features is divided into n clusters and then the distance of each frame of training voice
      sample from its nearest cluster is calculated,
''' 
def test_for_recognition():
    result=[]        #temp list that will contain information regarding the identity of the test voice sample
    for count in range(30):   #this loop will run for 30 times just to increase the accuracy of the system
        training_mfcc_feat=[]  #list that will hold mfcc_feature of training voice sample
        testing_mfcc_feat=[]   #list that will hold mfcc_feature of testing voice sample
        templist=[]
        templist.clear()
        training_mfcc_feat=np.array(mfcc.findmfcc(file[0]))
        training_mfcc_feat=sc.fit_transform(training_mfcc_feat) #scaling the mfcc feature
        estimator=KMeans(n_clusters=128)      
        testing_mfcc_feat=np.array(mfcc.findmfcc(file[1]))
        testing_mfcc_feat=sc.fit_transform(testing_mfcc_feat)
        if len(training_mfcc_feat)>len(testing_mfcc_feat):
            length=len(testing_mfcc_feat)
        else:
            length=len(training_mfcc_feat)    
        training_mfcc_feat=training_mfcc_feat[:length]    #making the mfcc features of training voice sample
                                                          #and testing voice sample of same length
        testing_mfcc_feat=testing_mfcc_feat[:length]
        estimator.fit(training_mfcc_feat)
        t=estimator.cluster_centers_       #t contains the information of each cluster center
        tq=vq(training_mfcc_feat,t)     #tq contains the distance of each frame of training voice sample from its 
                                        #nearest cluster where clusters are formed from the training voice sample.
        v=vq(testing_mfcc_feat,t)       #v contains the distance of each frame of testing voice sample from its 
                                        #nearest cluster where clusters are formed from the training voice sample
        distance=0.0         #temp variable that will hold the euclidean distance between each frame of v and tq.
        for row in range(length):
            distance += (v[1][row]-tq[1][row])**2
                        
        templist.append(sqrt(distance))
        result.append(min(templist))
        
    return result  
############################################### 

############################################### 
'''
FUNCTION NAME:databasetest
INPUT:NONE
OUTPUT:list containing the idea regarding the identity of test voice sample.
LOGIC:Each training voice sample is trained seperately and vq is applied between each training voice and
      testing voice, Each mfcc_features is divided into n clusters and then the distance of each frame from 
      its nearest cluster is calculated.
''' 
def databasetest():
    result=[]         #temp list that will contain information regarding the identity of the test voice sample
    for count in range(20):    #this loop will run for 30 times just to increase the accuracy of the system
        mfcc_feat=[]        #list that will hold the mfcc feature of each voice sample.
        trained_data=[]
        newtraineddata=[]   
        templist=[]    
        for speaker in range(len(file)):   
            mfcc_feat.append(sc.fit_transform(np.array(mfcc.findmfcc(file[speaker]))))
        length=min(map(len,mfcc_feat))
        for i in range(len(mfcc_feat)):
            mfcc_feat[i]=mfcc_feat[i][:length]    
        for i in range(len(mfcc_feat)-1):        
            trained_data.append(train(mfcc_feat[i]))
        for i in range(len(mfcc_feat)-1):
            newtraineddata.append(vq(mfcc_feat[-1],trained_data[i][0]))
        for i in range(len(mfcc_feat)-1):
            distance = 0.0
            for j in range(length):                
                distance += (trained_data[i][1][1][j]-newtraineddata[i][1][j])**2  #finding euclidean distance              
            templist.append(sqrt(distance))
        result.append(templist)
    result=calculate(result) 
    return result
###############################################    