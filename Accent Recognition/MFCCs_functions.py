import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import wave
import contextlib
from time import time
import csv
import pandas as pd
import os
from random import randrange
import librosa, librosa.display
from tensorflow.keras import layers, models, losses
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from python_speech_features import mfcc
import copy 
from sklearn.utils import shuffle

import pydot
import pydotplus
from pydotplus import graphviz

#Fixed the librosa problem :) B

import glob
import copy

def get_MFCCs():
    print('you will get the MFCCs in a dictionnary. It could take some time (around 10min)')
    debut = time()
    MFCCs_dict={}
    MFCCs_path = '/home/jovyan/work/MFCCs/'
    csv_file = glob.glob(MFCCs_path+'*.csv')
    decompte = len(csv_file)
    for filename in csv_file:
        print(decompte)
        speaker = os.path.basename(filename)
        speaker = speaker[:len(speaker)-4]
        df=pd.read_csv(filename)
        MFCCs_dict[speaker] = df.to_numpy()[:,1:]
        decompte-=1

    fin = time()
    print(fin-debut)
    return MFCCs_dict

def get_Speaker(Languages,MFCCs_dict={},Comment = False):

    Speaker = {}
    results = {}
    for language in Languages :
        results[language]=glob.glob('data_csv/'+language+'.csv')


    with open('data_csv/english.csv') as csv_file:
        csv_reader = pd.read_csv(csv_file, delimiter=',')
        if Comment :
            print('The columns of the file are : '+ str(list(csv_reader.columns)))
        for row in range(len(csv_reader)):
            filename = csv_reader['filename'][row]
            if len(MFCCs_dict)>0:
                if filename in MFCCs_dict.keys():
                    sex = csv_reader['sex'][row]
                    country = csv_reader['residence'][row]
                    Speaker[filename]={'gender' : sex,'agm_90' : filename + '_90','agm_110' : filename + '_110'}
                    Speaker[filename]['label'] = 1                
                    
            else :
                sex = csv_reader['sex'][row]
                country = csv_reader['residence'][row]
                Speaker[filename]={'gender' : sex,'agm_90' : filename + '_90','agm_110' : filename + '_110'}
                Speaker[filename]['label'] = 1                
                

        if Comment :        
            print('Processed {row} lines.')
            files_augment = list(Speaker.keys())
            labels_augment = [Speaker[filename]['label'] for filename in Speaker.keys()]
            print(labels_augment)
            genders_augment = [Speaker[filename]['gender'] for filename in Speaker.keys()]
            print(genders_augment)
            print("Same size ? " + str(len(files_augment) == len(labels_augment)),'\n')
            print("English")
            print(genders_augment.count('male'))
            print(genders_augment.count('female'))

    # Second case : For the other languages. All the label are equal to 0

    for language in Languages :
        if language != 'english':
            print(language)
            if Comment :
                index_begin_language = len(labels_augment)
            for csv_file in results[language]:
                print(csv_file)
                csv_reader = pd.read_csv(csv_file,sep = ',')
            
                print('The columns of the file are : ' + str(list(csv_reader.columns)))
                for row in range(len(csv_reader)):
                    filename = csv_reader['filename'][row]
                    if len(MFCCs_dict)>0:
                        if filename in MFCCs_dict.keys():
                            sex = csv_reader['sex'][row]
                            
                            Speaker[filename]={'gender' : sex, 'label': 0,'agm_90' : filename + '_90','agm_110' : filename + '_110'}
                            if Comment :
                                files_augment.append(filename)
                                genders_augment.append(sex)
                                labels_augment.append(0)
                    else : 
                        if filename in MFCCs_dict.keys():
                            sex = csv_reader['sex'][row]
                            
                            Speaker[filename]={'gender' : sex, 'label': 0,'agm_90' : filename + '_90','agm_110' : filename + '_110'}
                            if Comment :
                                files_augment.append(filename)
                                genders_augment.append(sex)
                                labels_augment.append(0)
            if Comment :
                print(f'Processed {row} lines.')
                print(files_augment[index_begin_language:])
                print(labels_augment[index_begin_language:])
                print(genders_augment[index_begin_language:])
                print("Same size ? " + str(len(files_augment) == len(labels_augment)),'\n')
                print(language)
                print(genders_augment[index_begin_language:].count('male'))
                print(genders_augment[index_begin_language:].count('female'))

    if Comment :       
        print('All the files : '+ str(len(files_augment)))
        print('All files augmented_90 : '+ str([Speaker[filename]['agm_90'] for filename in Speaker.keys()]))
        print('All the labels : '+ str(labels_augment))
        print('All the gender : '+str(genders_augment))
        print("Same size ? " + str(len(files_augment) == len(labels_augment)),'\n')

    return Speaker

def shuffle_MFCCs(Speaker,Comment = False):
    testSet_len = int(0.1 * len(Speaker))

    while testSet_len % 4 != 0:    
        testSet_len += 1

    male_count1 = female_count1 = male_count0 = female_count0 = 0
    gender_len = testSet_len/4

    permut = np.random.permutation(len(Speaker))
    files = list(Speaker.keys())
    if Comment :
        print(len(Speaker))
        print('Size of the test set (without considering augmentation : ', testSet_len)
        


    for i in permut :
        filename = files[i]

        if Speaker[filename]['label'] == 1 and Speaker[filename]['gender'] == 'male' and male_count1 < gender_len:
            Speaker[filename]['set']='test'
            male_count1+=1
        elif Speaker[filename]['label'] == 1 and Speaker[filename]['gender'] == 'female' and female_count1 < gender_len:
            Speaker[filename]['set']='test'
            female_count1+=1
        elif Speaker[filename]['label'] == 0 and Speaker[filename]['gender'] == 'male' and male_count0 < gender_len:
            Speaker[filename]['set']='test'
            male_count0+=1
        elif Speaker[filename]['label'] == 0 and Speaker[filename]['gender'] == 'female' and female_count0 < gender_len:
            Speaker[filename]['set']='test'
            female_count0+=1
        else:
            Speaker[filename]['set']='training'

    if Comment :
        print('Number of Native male in the test : ',male_count1)
        print('Number of Native female in the test : ',female_count1)
        print('Number of Non-Native female in the test : ',male_count0)
        print('Number of Non-Native female in the test : ',female_count0)

    print('Your training and testing set were shuffled again')


def get_train_test_set(MFCCs_dict,Speaker,Augmentation=False,Cutting=False,Comment=False):
    MFCCs_list_train = []
    labels_list_train = []
    
    MFCCs_list_test = []
    labels_list_test = []
    if Augmentation:
        print("You are recuperate the data with augmentation")
    for filename in Speaker.keys():
        if filename == "english501":
            continue

        if Speaker[filename]['set']=='training':
            
            MFCCs_list_train.append(MFCCs_dict[filename])
            labels_list_train.append(Speaker[filename]['label'])
            
            if Augmentation :
                filename_90 = Speaker[filename]['agm_90']
                filename_110 = Speaker[filename]['agm_110']
                
                MFCCs_list_train.append(MFCCs_dict[filename_90])
                labels_list_train.append(Speaker[filename]['label'])
                
                MFCCs_list_train.append(MFCCs_dict[filename_110])
                labels_list_train.append(Speaker[filename]['label'])
        else :
        
            MFCCs_list_test.append(MFCCs_dict[filename])
            labels_list_test.append(Speaker[filename]['label'])

            if Augmentation :
                filename_90 = Speaker[filename]['agm_90']
                filename_110 = Speaker[filename]['agm_110']

                MFCCs_list_test.append(MFCCs_dict[filename_90])
                labels_list_test.append(Speaker[filename]['label'])
                
                MFCCs_list_test.append(MFCCs_dict[filename_110])
                labels_list_test.append(Speaker[filename]['label'])
    
    
    if Cutting :
        print('MFCCs are cutting into a size equivalent to 4 secondes (400 value)')
        N = 2999 # length of MFCCs
        cut = 400

        new_MFCCs_list_train=[]
        new_labels_list_train=[]
        for i in range(len(MFCCs_list_train)):
            step = 0
            while (step+1)*cut < N:
                new_MFCCs_list_train.append(MFCCs_list_train[i][:,step*cut:(step+1)*cut])
                new_labels_list_train.append(labels_list_train[i])
                step+=1
        
        new_MFCCs_list_test=[]
        new_labels_list_test=[]
        for i in range(len(MFCCs_list_test)):
            step = 0
            while (step+1)*cut < N:
                new_MFCCs_list_test.append(MFCCs_list_test[i][:,step*cut:(step+1)*cut])
                new_labels_list_test.append(labels_list_test[i])
                step+=1
        if Comment :
            print( 'Size of training data : ',len(new_MFCCs_list_train))
            print('Size of the labels of the training data : ',len(new_labels_list_train))

            print('Size of test data : ',len(new_MFCCs_list_test))
            print('Size of the labels of the training data : ',len(new_labels_list_test))

            print('Shape of MFCCs : ', new_MFCCs_list_test[0].shape)
        return((new_MFCCs_list_train,new_labels_list_train),(new_MFCCs_list_test,new_labels_list_test))

    else : 
        if Comment :
            print( 'Size of training data : ',len(MFCCs_list_train))
            print('Size of the labels of the training data : ',len(labels_list_train))

            print('Size of test data : ',len(MFCCs_list_test))
            print('Size of the labels of the training data : ',len(labels_list_test))
            print('Shape of MFCCs : ', MFCCs_list_test[0].shape)
        return((MFCCs_list_train,labels_list_train),(MFCCs_list_test,labels_list_test))

def shuffle_set(MFCCs_list,labels_list):
    permut = np.random.permutation(len(MFCCs_list))
    permut_MFCCs_list= copy.deepcopy(MFCCs_list)
    permut_labels_list = copy.deepcopy(labels_list)

    for i in range(len(MFCCs_list)) :
        MFCCs_list[i] = permut_MFCCs_list[permut[i]]
        labels_list[i] = permut_labels_list[permut[i]]



def calculate_MFCCs(Speaker,Augmentation = False, MFCCs_dict={}):
    Liste_key=Speaker.keys()
    for i, f in enumerate(Liste_key):
        
        filename= glob.glob('/datasets/*/'+f+'.wav')
        
        print(filename)
        if len(filename)>1:
            del Speaker[f]
            continue
        elif len(filename)==0:
            print(f)
            del Speaker[f]
            continue
        else : # Good case
            if f not in  MFCCs_dict:
                MFCCs_dict[f] = make_normed_mfcc(filename[0], outrate = 8000)
                if Augmentation :
                    f_90 = Speaker[f]['agm_90']
                    f_110 = Speaker[f]['agm_110']
                    
                    filename_90 = glob.glob('/datasets/augmented-dataset/'+f_90+'.wav')[0]
                    print(filename_90)
                    filename_110 = glob.glob('/datasets/augmented-dataset/'+f_110+'.wav')[0]
                    print(filename_110)
                    
                    MFCCs_dict[f_90]=make_normed_mfcc(filename_90, outrate = 8000)
                    MFCCs_dict[f_110]=make_normed_mfcc(filename_110, outrate = 8000)

def savecsv_MFCCs(MFCCs_dict):
    dos_path = '/home/jovyan/work/MFCCs/'
    for speaker in MFCCs_dict.keys():
        print(speaker)
        csv_path = dos_path + speaker + '.csv'
        df = pd.DataFrame(MFCCs_dict[speaker])  
        df.to_csv(csv_path, sep = ',') 


# read in signal, change sample rate to outrate (samples/sec), use write_wav=True to save wav file to disk
def downsample(filename, outrate=8000, write_wav = False):
    #(rate, sig) = wav.read(filename)
    (sig, rate) = librosa.load(filename, sr=outrate)
    down_sig = librosa.core.resample(sig, rate, outrate, scale=True)
    if not write_wav:
        return down_sig, outrate
    if write_wav:
        wav_write('{}_down_{}.wav'.format(filename, outrate), outrate, down_sig)

# change total number of samps for downsampled file to n_samps by trimming or zero-padding and standardize them
def make_standard_length(filename, n_samps=240000):
    down_sig, rate = downsample(filename)
    normed_sig = librosa.util.fix_length(down_sig, n_samps)
    normed_sig = (normed_sig - np.mean(normed_sig))/np.std(normed_sig)
    return normed_sig

# for input wav file outputs (13, 2999) mfcc np array
def make_normed_mfcc(filename, outrate=8000):
    normed_sig = make_standard_length(filename)
    normed_mfcc_feat = mfcc(normed_sig, outrate)
    normed_mfcc_feat = normed_mfcc_feat.T
    #normed_mfcc_feat = librosa.feature.mfcc(normed_sig, sr = outrate, n_mfcc = 13)
    return normed_mfcc_feat
