import os
import numpy as np
import librosa
import time

#%% Get training data
freebird_path = "C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment2/freefield1010/ff1010bird_wav"
warbird_path = "C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment2/warblrb10k"

spectrograms = []
start = time.time()
i = 0
for path in [freebird_path]:
    for file in os.listdir(path):
        if file.split(".")[1] != "wav":
            continue
        filepath = path + "/" + file
        audio_data  = librosa.load(filepath, res_type='kaiser_best', sr=44100)[0]
        i+=1
        audio_data = audio_data * 1/np.max(np.abs(audio_data))
        kwargs_for_mel = {'n_mels': 40}
        spectrogram = librosa.feature.melspectrogram(y=audio_data,
                                               sr=44100, 
                                               n_fft=1024, 
                                               hop_length=512, 
                                               **kwargs_for_mel) 
        spectrograms.append(spectrogram)
        if i in np.arange(0, 20000, 500):
            print(i)
        
    

print(f"Time for reading the data: {((time.time()-start)):.2f} s")
np.save("freefield1010-spectrograms", spectrograms)


#%% Get test data
start = time.time()
kwargs_for_mel = {'n_mels': 40}
direc = "C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment2/"
spectrograms_test = []
i=0
for file in os.listdir(direc+"/test"):
    if file.split(".")[1] != "npy":
        continue
    filepath = direc + "test/" + file
    audio_data  = np.load(filepath)
    audio_data = librosa.resample(audio_data, 48000, 44100)
    audio_data = audio_data * 1/np.max(np.abs(audio_data))
    kwargs_for_mel = {'n_mels': 40}
    spectrogram = librosa.feature.melspectrogram(y=audio_data,
                                           sr=44100, 
                                           n_fft=1024, 
                                           hop_length=512, 
                                           **kwargs_for_mel) 
    spectrograms_test.append(spectrogram)
    i+=1
    if i in np.arange(0, 20000, 500):
        print(i)
    
print(f"Time for reading the data: {((time.time()-start)):.2f} s")
np.save("test-spectrograms", spectrograms_test)


#%%
direc = "C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment2/"
y = np.load(direc+"freefield1010-labels.npy").astype(int)
X = np.swapaxes(np.load(direc+"freefield1010-spectrograms.npy"), 1, 2)[..., np.newaxis]

#%%
from sklearn.model_selection import train_test_split
# Muodostetaan 70-20-10 train-valid-test, joiden jakaumat vastaavat alkuperäisen
# joukon jakaumaa (stratify).
# Harjoitus-validointi split
trainX, validX, trainY_input, validY_input = train_test_split(X, 
                                                              y, 
                                                              test_size=0.3,
                                                              random_state=1,
                                                              stratify=y)
# Luodaan vielä erillinen testijoukko
validX, testX, validY_input, testY_input = train_test_split(validX,
                                                            validY_input,
                                                            test_size=0.33,
                                                            random_state=1,
                                                            stratify=validY_input)


#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

filters = 64
dropout_rate = 0.5
model = Sequential()

model.add(Conv2D(filters, kernel_size=5, activation='relu', padding="same",
                 input_shape=(862, 40, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((1, 5)))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))

model.add(Conv2D(filters, kernel_size=5, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((1, 2)))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))

model.add(Conv2D(filters, kernel_size=5, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((1, 2)))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))

model.add(Conv2D(filters, kernel_size=5, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((1, 2)))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(96, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid'))
#%% Compile model

epochs = 100
batch_size = 128
learning_rate = 0.00001
print(f"Running {epochs} epochs with batch size of {batch_size} and learning rate of {learning_rate}.")
opt = tf.optimizers.Adam(lr=learning_rate)

# Keskeyttää fitin, jos validation AUC ei parane 10 epochiin.
callback = EarlyStopping(monitor='val_auc', patience=10, 
                         restore_best_weights=True, mode="max")
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['AUC', 'accuracy'])


#%% Without data augmentation
history = model.fit(trainX, trainY_input, 
                    epochs=epochs, 
                    validation_data=(validX, validY_input), 
                    batch_size=batch_size, 
                    callbacks=[callback],
                    verbose=2)

print("Finished")