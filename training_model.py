import scipy.io.wavfile as wav
import numpy as np 
from sklearn.preprocessing import StandardScaler
from random import seed
from random import randrange
from math import sqrt
from sklearn.externals import joblib
import mfcc
sc=StandardScaler()
distance=[]
x=[]
y=[]
y.clear()
z=[]
codeboo=[]
#####################################################
mfcc_feat1=[]
rate,sig=wav.read('rk1.wav')
mfcc_feat1=mfcc.findmfcc([rate,sig])
mfcc_feat1=sc.fit_transform(mfcc_feat1)        
mfcc_feat1=mfcc_feat1.tolist()
for i in range(len(mfcc_feat1)):
    mfcc_feat1[i].append(0)

mfcc_feat3=[]
rate,sig=wav.read('rk2.wav')
mfcc_feat3=mfcc.findmfcc([rate,sig])
mfcc_feat3=sc.fit_transform(mfcc_feat3)        
mfcc_feat3=mfcc_feat3.tolist()
for i in range(len(mfcc_feat3)):
    mfcc_feat3[i].append(0)

mfcc_feat5=[]
rate,sig=wav.read('an3.wav')
mfcc_feat5=mfcc.findmfcc([rate,sig])
mfcc_feat5=sc.fit_transform(mfcc_feat5)        
mfcc_feat5=mfcc_feat5.tolist()
for i in range(len(mfcc_feat5)):
    mfcc_feat5[i].append(1)
    
mfcc_feat6=[]
rate,sig=wav.read('an4.wav')
mfcc_feat6=mfcc.findmfcc([rate,sig])
mfcc_feat6=sc.fit_transform(mfcc_feat6)        
mfcc_feat6=mfcc_feat6.tolist()
for i in range(len(mfcc_feat6)):
    mfcc_feat6[i].append(1)
 
mfcc_feat=mfcc_feat3+mfcc_feat1+mfcc_feat5+mfcc_feat6  
#######################################################
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(12):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
######################################################
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
def get_best_matching_unit(codebooks, test_row):
    distances = list()
    distance.clear()
    for codebook in codebooks:
        dist = euclidean_distance(codebook, test_row)
        distances.append((codebook, dist))
        distance.append(dist)
    distances.sort(key=lambda tup: tup[1])
    return distances[0][0]
####################################################
# Make a prediction with codebook vectors
def predict(codebooks, test_row):
    x.clear()
    z.clear()
    x.append(get_best_matching_unit1(codebooks, test_row))
    z.append(x[0][0][-1])
    z.append(x[0][1])
    return z
###################################################
# Create a random codebook vector
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook
###################################################
# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epochs):
    codebooks = [random_codebook(mfcc_feat1) for i in range(n_codebooks)]+[random_codebook(mfcc_feat3) for i in range(n_codebooks)]+[random_codebook(mfcc_feat5) for i in range(n_codebooks)]+[random_codebook(mfcc_feat6) for i in range(n_codebooks)]    #+[random_codebook(mfcc_feat4) for i in range(n_codebooks)]+[random_codebook(mfcc_feat6) for i in range(n_codebooks)]
    for epoch in range(epochs):
        rate = lrate * (1.0-(epoch/float(epochs)))
        for row in train:
            bmu = get_best_matching_unit(codebooks, row)
            for i in range(len(row)-1):
                error = row[i] - bmu[i]
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
    y.append(codebooks)
    return codebooks
#################################################
seed(1)
codebook=train_codebooks(mfcc_feat,60,0.55,63)
np.save('vc10-15587',codebook)
filename='fi.sav'
joblib.dump(sc,filename)