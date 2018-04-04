from tkinter import filedialog        #for opening file explorer
from tkinter import font              #for changing font of the words
from tkinter import *
import numpy as np  
import matplotlib.pyplot as plt                  
import final                          #python file implementing lvq
import vectquan                       #python file implementing vq
import scipy.io.wavfile               #for reading wav file
import sounddevice as sd              #for recording and playing
import pyttsx3                        #for voice
######################################################
#This set the voice which will be used later
engine=pyttsx3.init()
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
#####################################################
order=['First','Second','Third','Fourth','fifth','sixth','seventh','eigth','ninth','tenth']
root=Tk()                            #for blank window
root.title('main window')            #for giving title to the window
vectquan.file.clear()                #this function clear the list named as file in vectquan python file

####################################################
'''
FUNCTION NAME:training_for_recognition_by_recording
INPUT:NONE
OUTPUT:NONE
LOGIC:This is the function which is used to train the vq model for speaker recognition system by giving the user
      an option of recording their voice sample and then append that file into the list 
      named as file located in vectquan python file.
'''
def training_for_recognition_by_recording():
    templist=[]    #this is the temporary list which will hold the sample rate and signal array of the wav file
    vectquan.file.clear()
    sample_rate=8000
    duration=5
    signal=sd.rec(int(duration*sample_rate),samplerate=sample_rate,channels=1)
    sd.wait()
#    filename='new'+str(1)+'.wav'
#    librosa.output.write_wav(filename,myrecording,sr=8000)
    templist.append(sample_rate)
    templist.append(signal)
    vectquan.file.append(templist) 
####################################################

####################################################
'''
FUNCTION NAME:testing_for_recognition_by_recording
INPUT:NONE
OUTPUT:Give the result whether the voice matches or not.
LOGIC:This is the function which is used to test the vq model for speaker recognition system by giving the user
      an option of recording their voice sample and then append that file into the list 
      named as file located in vectquan python file.
''' 
def testing_for_recognition_by_recording():
    templist=[]    #this is the temporary list which will hold the sample rate and signal array of the wav file
    sample_rate=8000
    duration=5
    signal=sd.rec(int(duration*sample_rate),samplerate=sample_rate,channels=1)
    sd.wait()
#    filename='new'+str(1)+'.wav'
#    librosa.output.write_wav(filename,myrecording,sr=8000)

    templist.append(sample_rate)
    templist.append(signal)
    vectquan.file.append(templist)      #the particular information of the user that is sample rate and
                                        #signal array is appended to a list which is located in vectquan python file.
    validity=vectquan.test_for_recognition()  #validity is the list that hold the values which will give idea whether
                                              #the voice sample matches or not.
    if (sum(validity)/len(validity))<30.5:
        print('voice matched')
        engine.say('voice matched')
        engine.setProperty('rate',120)
        engine.setProperty('volume',0.9)
        engine.runAndWait()
    else:
        print('voice does not matched')
        engine.say('voice doesnot matched')
        engine.setProperty('rate',120)
        engine.setProperty('volume',0.9)
        engine.runAndWait()
    vectquan.file.clear()               #the list is cleared for another operation.
#################################################### 
  
####################################################
'''
FUNCTION NAME:training_for_recognition
INPUT:NONE
OUTPUT:NONE
LOGIC:This is the function which is used to train the vq model for speaker recognition system by giving the user
      an option of selecting any wave file by opening the file explorer and then append that file into the list 
      named as file located in vectquan python file.
'''
def training_for_recognition():
    templist=[]    #this is the temporary list which will hold the sample rate and signal array of the wav file
    vectquan.file.clear()
    root.filename=filedialog.askopenfilename(initialdir="/",title='select file',filetypes=(('wave files','*.wav'),('all files','*.*')))
    print(root.filename)
    sample_rate, signal = scipy.io.wavfile.read(root.filename)    
    sd.play(signal,sample_rate)
    try:
        signal=signal[:,0]
    except:
        signal=signal
    plt.specgram(signal[4000:],Fs=sample_rate)
    plt.title('spectrogram of voice sample')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.show()
    templist.append(sample_rate)
    templist.append(signal)
    vectquan.file.append(templist)        #the particular information of the user that is sample rate and
                                          #signal array is appended to a list which is located in vectquan python file.
    
####################################################

####################################################
'''
FUNCTION NAME:testing_for_recognition
INPUT:NONE
OUTPUT:Give the result whether the voice matches or not.
LOGIC:This is the function which is used to test the vq model for speaker recognition system by giving the user
      an option of selecting any wave file by opening the file explorer and then append that file into the list 
      named as file located in vectquan python file.
'''    
def testing_for_recognition():
    templist=[]    #this is the temporary list which will hold the sample rate and signal array of the wav file.
    root.filename=filedialog.askopenfilename(initialdir="/",title='select file',filetypes=(('wave files','*.wav'),('all files','*.*')))
    print(root.filename)
    sample_rate, signal = scipy.io.wavfile.read(root.filename)
    sd.play(signal,sample_rate)
    sd.wait()
    try:
        signal=signal[:,0]
    except:
        signal=signal
    plt.specgram(signal[4000:],Fs=sample_rate)
    plt.title('spectrogram of voice sample')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.show()
    templist.append(sample_rate)
    templist.append(signal)
    vectquan.file.append(templist)      #the particular information of the user that is sample rate and
                                        #signal array is appended to a list which is located in vectquan python file.
    validity=vectquan.test_for_recognition()  #validity is the list that hold the values which will give idea whether
                                              #the voice sample matches or not.
    if (sum(validity)/len(validity))<30.5:
        print('voice matched')
        engine.say('voice matched')
        engine.setProperty('rate',120)
        engine.setProperty('volume',0.9)
        engine.runAndWait()
    else:
        print('voice does not matched')
        engine.say('voice doesnot matched')
        engine.setProperty('rate',120)
        engine.setProperty('volume',0.9)
        engine.runAndWait()
    vectquan.file.clear()               #the list is cleared for another operation.
###################################################### 

######################################################
'''
FUNCTION NAME:training_for_identification_by_recording
INPUT:NONE
OUTPUT:NONE
LOGIC:This is the function which is used to train the vq model for speaker identification system by giving the user
      an option of recording their voice sample and then append that file into the list 
      named as file located in vectquan python file.The only difference between training_for_recognition function
      and training_for_identification function is the no.of voice samples used for training,In training_for_identification
      the training voice may be one or more whereas in training_for_recognition the training is done with only 
      one voice samples.
'''
def training_for_identification_by_recording():
    templist=[]    #this is the temporary list which will hold the sample rate and signal array of the wav file
    sample_rate=8000
    duration=5
    signal=sd.rec(int(duration*sample_rate),samplerate=sample_rate,channels=1)
    sd.wait()
#    filename='new'+str(1)+'.wav'
#    librosa.output.write_wav(filename,myrecording,sr=8000)
    templist.append(sample_rate)
    templist.append(signal)
    vectquan.file.append(templist)
####################################################

####################################################
'''
FUNCTION NAME:testing_for_identification_by_recording
INPUT:NONE
OUTPUT:Give the result whether the voice matches with any of the trained voices and if it matches then with which 
       voice sample it matches the most.
LOGIC:This is the function which is used to test the vq model for speaker identification system by giving the user
      an option of recording their voice sample and then append that file into the list named as file  
      located in vectquan python file.
'''  
def testing_for_identification_by_recording():
    templist=[]    #this is the temporary list which will hold the sample rate and signal array of the wav file
    sample_rate=8000
    duration=5
    signal=sd.rec(int(duration*sample_rate),samplerate=sample_rate,channels=1)
    sd.wait()
#    filename='new'+str(1)+'.wav'
#    librosa.output.write_wav(filename,myrecording,sr=8000)
    templist.append(sample_rate)
    templist.append(signal)
    vectquan.file.append(templist)
    validity=vectquan.databasetest()    #validity is the list that hold the values which will give idea whether
                                        #the voice sample matches or not.  
    if min(validity)>30.5:
        print('voice  doesnot matched with any stored voices')
        engine.say('voice  does not matched with any stored voices')
        engine.setProperty('rate',120)
        engine.setProperty('volume',0.9)
        engine.runAndWait()
    else:
        print('voice matched  more with '+order[np.argmin(validity)]+' person')
        engine.say('voice matched more with '+order[np.argmin(validity)]+' person')
        engine.setProperty('rate',120)
        engine.setProperty('volume',0.9)
        engine.runAndWait()
######################################################
    
######################################################
'''
FUNCTION NAME:training_for_identification
INPUT:NONE
OUTPUT:NONE
LOGIC:This is the function which is used to train the vq model for speaker identification system by giving the user
      an option of selecting any wave file by opening the file explorer and then append that file into the list 
      named as file located in vectquan python file.The only difference between training_for_recognition function
      and training_for_identification function is the no.of voice samples used for training,In training_for_identification
      the training voice may be one or more whereas in training_for_recognition the training is done with only 
      one voice samples.
'''
def training_for_identification():
    templist=[]        #this is the temporary list which will hold the sample rate and signal array of the wav file.
    root.filename=filedialog.askopenfilename(initialdir="/",title='select file',filetypes=(('wave files','*.wav'),('all files','*.*')))
    print(root.filename)
    sample_rate, signal = scipy.io.wavfile.read(root.filename)
    sd.play(signal,sample_rate)
    #sd.wait()
    templist.append(sample_rate)
    templist.append(signal)
    vectquan.file.append(templist) 
####################################################
    
####################################################
'''
FUNCTION NAME:testing_for_identification
INPUT:NONE
OUTPUT:Give the result whether the voice matches with any of the trained voices and if it matches then with which 
       voice sample it matches the most.
LOGIC:This is the function which is used to test the vq model for speaker identification system by giving the user
      an option of selecting any wave file by opening the file explorer and then append that file into the list 
      named as file located in vectquan python file.
'''   
def testing_for_identification():
    templist=[]      #this is the temporary list which will hold the sample rate and signal array of the wav file.
    root.filename=filedialog.askopenfilename(initialdir="/",title='select file',filetypes=(('wave files','*.wav'),('all files','*.*')))
    print(root.filename)
    sample_rate, signal = scipy.io.wavfile.read(root.filename)
    sd.play(signal,sample_rate)
    sd.wait()
    templist.append(sample_rate)
    templist.append(signal)
    vectquan.file.append(templist)
    validity=vectquan.databasetest()    #validity is the list that hold the values which will give idea whether
                                        #the voice sample matches or not.  
    if min(validity)>30.5:
        print('voice  doesnot matched with any stored voices')
        engine.say('voice  does not matched with any stored voices')
        engine.setProperty('rate',120)
        engine.setProperty('volume',0.9)
        engine.runAndWait()
    else:
        print('voice matched  more with '+order[np.argmin(validity)]+' person')
        engine.say('voice matched more with '+order[np.argmin(validity)]+' person')
        engine.setProperty('rate',120)
        engine.setProperty('volume',0.9)
        engine.runAndWait()
###################################################

###################################################
'''
FUNCTION NAME:test_lvq_system
INPUT:NONE
OUTPUT:Give the result whether the voice matches with any of the trained voices and if it matches then with which 
       voice sample it matches the most.
LOGIC:This is the function which is used to test the lvq model for speaker identification system ,in this system
      the lvq model is trained with voice sample and then it gives the user an option of choosing any wave file from the 
      file explorer and test the lvq model.
'''  
def test_lvq_system():
    final.filename.clear()
    templist=[]    #this is the temporary list which will hold the sample rate and signal array of the wav file.
    root.filename=filedialog.askopenfilename(initialdir="/",title='select file',filetypes=(('wave files','*.wav'),('all files','*.*')))
    print(root.filename)
    sample_rate, signal = scipy.io.wavfile.read(root.filename)
    sd.play(signal,sample_rate)
    sd.wait()
    templist.append(sample_rate)
    templist.append(signal)
    final.filename.append(templist)
    result=final.find()
    if (max(result))<0.70:
        print('voice  doesnot matched with any stored voices')
        engine.say('voice  does not matched with any stored voices')
        engine.setProperty('rate',120)
        engine.setProperty('volume',0.9)
        engine.runAndWait()
    else:
        print('voice matched more with '+str(final.sequence[np.argmax(result)])+' speaker')
        engine.say('voice matched more with '+str(final.sequence[np.argmax(result)])+' speaker')
        engine.setProperty('rate',120)
        engine.setProperty('volume',0.9)
        engine.runAndWait()
###################################################

###################################################
'''
FUNCTION NAME:leave
INPUT:NONE
OUTPUT:NONE
LOGIC:IT Simply destroy or closes the windows/GUI
'''
def leave():
    print('thank you') 
    root.destroy()
################################################### 

###################################################
#BELOW CODE WILL MAKE THE GUI AS PER THE REQUIREMENT       
font=font.Font(family='Helvetica',size=14,weight='bold')     #it assign the font variable with the information of the font.
label1=Label(root,text='GUI FOR SPEAKER VERIFICATION SYSTEM')#this is the label which gives the idea about what 
                                                             #the gui is used for 
label1['font']=font                                          #it set the font of the letters displayed in the GUI
label1.grid(row=0,column=0)

frame=Frame(root,width=15,height=15)
frame.grid(row=1,column=0)

label2=Label(root,text='HOW DO YOU WANT TO TEST THE SPEAKER VERIFICATION SYSTEM:',fg='red')    
label2.grid(row=2,column=0)

frame=Frame(root,width=15,height=15)
frame.grid(row=3,column=0)

label3=Label(root,text='BY RECORDING:')    
label3.grid(row=4,column=0,sticky=E)

button1=Button(root,text='   training   ',command=training_for_recognition_by_recording)
button1.grid(row=4,column=1,sticky=W)
button2=Button(root,text='testing',command=testing_for_recognition_by_recording)
button2.grid(row=4,column=2,sticky=W)

frame=Frame(root,width=20,height=20)
frame.grid(row=5,column=0)

label4=Label(root,text='BY OPENING THE STORED FILE FROM FILE MANAGER:')    
label4.grid(row=6,column=0,sticky=E)

button3=Button(root,text=' select file  ',command=training_for_recognition)
button3.grid(row=6,column=1,sticky=W)
button4=Button(root,text='testing',command=testing_for_recognition)
button4.grid(row=6,column=2,sticky=W)

frame=Frame(root,width=20,height=20)
frame.grid(row=7,column=0)

label5=Label(root,text='HOW DO YOU WANT TO TEST THE SPEAKER IDENTIFICATION SYSTEM:',fg='red')    
label5.grid(row=8,column=0)

frame=Frame(root,width=20,height=20)
frame.grid(row=9,column=0)

label6=Label(root,text='BY RECORDING:')    
label6.grid(row=10,column=0,sticky=E)

button5=Button(root,text='  training    ',command=training_for_identification_by_recording)
button5.grid(row=10,column=1,sticky=W)
button6=Button(root,text='testing',command=testing_for_identification_by_recording)
button6.grid(row=10,column=2,sticky=W)

frame=Frame(root,width=20,height=20)
frame.grid(row=11,column=0)

label7=Label(root,text='BY OPENING THE STORED FILE FROM FILE MANAGER:')    
label7.grid(row=12,column=0,sticky=E)

button7=Button(root,text=' select file  ',command=training_for_identification)
button7.grid(row=12,column=1,sticky=W)
button8=Button(root,text='testing',command=testing_for_identification)
button8.grid(row=12,column=2,sticky=W)

frame=Frame(root,width=20,height=20)
frame.grid(row=13,column=0)

label8=Label(root,text='SPEAKER IDENTIFICATION SYSTEM USING LVQ:')          
label8.grid(row=14,column=0,sticky=E)

button8=Button(root,text='play stored\n voices',command=final.play_voice_sample_of_each_speaker)
button8.grid(row=14,column=1,sticky=W)
button9=Button(root,text='testing',command=test_lvq_system)
button9.grid(row=14,column=2,sticky=W)

frame=Frame(root,width=20,height=20)
frame.grid(row=15,column=0)

label9=Label(root,text='DO YOU WANT TO CLOSE THE GUI, PRESS THE QUIT BUTTON:')    
label9.grid(row=16,column=0,sticky=E)

button10=Button(root,text='QUIT',command=leave)
button10.grid(row=16,column=1,sticky=W)

frame=Frame(root,width=20,height=20)
frame.grid(row=17,column=0)

c=Canvas(root,width=400,height=300,bg='white')
c.grid(row=18,column=0) 
TEXT='There is two option given to each system i.e. either select a\n stored file and then check whether the system gives the desired \nresult or record your voice and do the testing..\n\nFor speaker verificaton system you have to record your \nvoice by pressing training button and then in order \nto test the system click on testing button and again record your voice,\nor choose the files from saved in to the memory.\nAnd for speaker identification many users have to record their voice \nby pressing the testing button and at the time of testing any\n user can come and record their voice by clicking on \ntesting voice and then record your voice or just use the \nselect from file option.'  
c.create_text(200,100,text=TEXT) 
engine.say('welcome to the graphical user interface for speaker identification and verification system')
engine.setProperty('rate',120)
engine.setProperty('volume',0.9)
engine.runAndWait()
root.mainloop()
################################################