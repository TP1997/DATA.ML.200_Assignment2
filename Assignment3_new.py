import os
import numpy as np
import librosa
import time

#%% Get training data
#freebird_path = "C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment2/freefield1010/ff1010bird_wav"
#warbird_path = "C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment2/warblrb10k"

freebird_path = "/home/tuomas/Python/DATA.ML.200/DATA.ML.200_Assignment2_Data/ff1010bird_train/wav/"
warbird_path = "/home/tuomas/Python/DATA.ML.200/DATA.ML.200_Assignment2_Data/warblrb10k_public_train/wav/"

#%%
#direc = "C:/Users/joona/OneDrive - TUNI.fi/PRML/Assignments/DATA.ML.200_Assignment2/"
direc = "/home/tuomas/Python/DATA.ML.200/Assignment2_data/"
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

