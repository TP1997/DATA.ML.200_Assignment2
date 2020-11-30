import os
import numpy as np
import librosa
import time
import pandas as pd

#%%
test_directory = '/home/tuomas/Python/DATA.ML.200/Assignment2_data/'

#%% Convert test data to spectogram
test_audio = np.load(test_directory+'test-audiodata.npy')
#%%
spectrograms = []
kwargs_for_mel = {'n_mels': 40}
n=0
for data in test_audio:
    spectrogram = librosa.feature.melspectrogram(y=data,
                                                sr=44100, 
                                                n_fft=1024, 
                                                hop_length=512, 
                                                power=1.,
                                                **kwargs_for_mel)
    spectrograms.append(spectrogram)
    
    n+=1
    if n in np.arange(0, 20000, 500): print(n)
    
np.save("test-spectrograms", spectrograms)