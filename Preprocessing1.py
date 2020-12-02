import os
import numpy as np
import librosa
import time
import pandas as pd
#%% Get training data
freebird_path = "C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment2/freefield1010/ff1010bird_wav/"
warbird_path = "C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment2/warblrb10k/"
test_path = r"C:\Users\joona\OneDrive - TUNI.fi\PRML\Assignments\DATA.ML.200_Assignment2\test/"
#freebird_path = "/home/tuomas/Python/DATA.ML.200/DATA.ML.200_Assignment2_Data/ff1010bird_train/"
#warbird_path = "/home/tuomas/Python/DATA.ML.200/DATA.ML.200_Assignment2_Data/warblrb10k_public_train/wav/"
#test_path = '/home/tuomas/Python/DATA.ML.200/DATA.ML.200_Assignment2_Data/test'

#%% Process test audio data (Downsampling only)
audio_data = []
directory = test_path
files = os.listdir(directory)
n=0
for file in files:
    data  = np.load(directory + '/' + file)
    data = librosa.resample(data, 48000, 44100)
    audio_data.append(data)
    
    n+=1
    if n in np.arange(0, 20000, 500): print(n)

np.save("test-audiodata", audio_data)
#%% Process freebird wav data to spectrograms
# Get audio data
spectrograms = []
directory = freebird_path 
files = os.listdir(directory)
n=0
kwargs_for_mel = {'n_mels': 40}
for file in files:
    data = librosa.core.load(directory+file, sr=44100, res_type='kaiser_best')[0]
    spectrogram = librosa.feature.melspectrogram(y=data,
                                                 sr=44100, 
                                                 n_fft=1024, 
                                                 hop_length=512, 
                                                 power=1.,
                                                 **kwargs_for_mel)
    db_spec = librosa.power_to_db(spectrogram, ref=np.max)
    spectrograms.append(db_spec.T)
    n+=1
    if n in np.arange(0, 20000, 500): print(n)
spectrograms = np.array(spectrograms)
np.save("freefield1010-spectrograms", spectrograms)

#%% Process freebird audio labels
directory = freebird_path
files = os.listdir(directory )
csv_name = 'ff1010bird_metadata_2018.csv'

ids = [int(fn[:-4]) for fn in files]
csv = pd.read_csv(directory+csv_name, sep=',')
csv_itemid = np.array(csv['itemid'].to_list())
csv_hasbird = np.array(csv['hasbird'].to_list())
print('Getting labels...')
labels=[ csv_hasbird[np.where(csv_itemid==id_)[0][0]] for id_ in ids]
        
np.save("freefield1010-labels", labels)

#%% Process warbird wav data to spectrograms
# Get audio data
spectrograms = []
directory = warbird_path 
files = os.listdir(directory)
target_length = 862
n=0
kwargs_for_mel = {'n_mels': 40}
for file in files:
    data = librosa.core.load(directory+file, sr=44100, res_type='kaiser_best')[0]
    spectrogram = librosa.feature.melspectrogram(y=data,
                                                 sr=44100, 
                                                 n_fft=1024, 
                                                 hop_length=512, 
                                                 power=1.,
                                                 **kwargs_for_mel) 
    length = spectrogram.shape[1]
    
    if length > target_length:
        #Crop from center
        start = length//2 - target_length//2
        end = start+target_length
        spectrogram = spectrogram[:, start:end]
    elif length < target_length:
        # Pad to target length
        spectrogram = librosa.util.pad_center(spectrogram, target_length, mode='constant')
    
    db_spec = librosa.power_to_db(spectrogram, ref=np.max)
    spectrograms.append(db_spec.T)
    
    n+=1
    if n in np.arange(0, 20000, 500): print(n)

spectrograms = np.array(spectrograms)
np.save("warblrb10k-spectrograms", spectrograms)