from scipy.io import wavfile
import numpy as np
import os, shutil
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import random


Notes = ['1_0.wav', '1_12.wav']
channels = ['1st channel', '2nd channel']

N = 128
notesLength = 3

trainSize = 0.8
testSize = 1-trainSize

table = PrettyTable()
column_names = [" Note ", " Time Duration ", " Channels ", " Sample Rate "]
table.title = ' Characteristics of Note '
table.field_names = [column_names[0], column_names[1], column_names[2], column_names[3]]
for i in range(len(Notes)):
    sampleRate, data = wavfile.read('./../notes/wav/'+Notes[i])
    for j in range(np.shape(np.array(data))[1]):
        plt.figure(figsize=(12, 5))
        plt.plot(data[:, j], linewidth=0.7)
        plt.ylabel("AMPLITUDE")
        plt.xlabel("SAMPLES")
        plt.title(channels[j]+' of '+Notes[i])
    table.add_row([Notes[i] ,str(1.0*np.shape(np.array(data))[0]/sampleRate) +' (s)' , np.shape(np.array(data))[1], str(sampleRate) + ' (samples/sec)'])

print(table,'\n')
plt.show()

folder = './../notes/dataset/'
for directory in os.listdir(folder):
    directory_path = os.path.join(folder, directory)
    for file in os.listdir(directory_path):
        wav_files_path = os.path.join(directory_path, file)
        try:
            if os.path.isfile(wav_files_path) or os.path.islink(wav_files_path):
                os.unlink(wav_files_path)
            elif os.path.isdir(wav_files_path):
                shutil.rmtree(wav_files_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (wav_files_path, e))

filenames = [str(j)+'_'+str(i)+'.wav' for j in range(N) for i in range(notesLength)]
random.shuffle(filenames)

dataset = {'Train':[], 'Test':[]}
dataset['Train'] = ([x for x in filenames[:int(trainSize*N*notesLength)]])
dataset['Test'] = ([x for x in filenames[int(trainSize*N*notesLength):]])

for datasetName in dataset:
    for fileName in dataset[datasetName]:
        shutil.copyfile('./../notes/wav/' + fileName, './../notes/dataset/' + datasetName + '/' + fileName)
