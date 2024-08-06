# %%
# %pip install librosa keras-preprocessing resample joblib

import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# %% [markdown]
# ## Loading and Displaying Audio Data

# %%
sample_audio_path = "./Audio_Speech_Actors_01-24/Actor_01"
sample_file = "03-01-05-02-01-02-01.wav"

# %%
# Load and display the sample audio waveform
data, sampling_rate = librosa.load(os.path.join(sample_audio_path, sample_file))
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data, sr=sampling_rate)
plt.show()

# %% [markdown]
# ### Calculate and display the spectrogram

# %%
sr, x = scipy.io.wavfile.read(os.path.join(sample_audio_path, sample_file))
nstep = int(sr * 0.01)
nwin = int(sr * 0.03)
nfft = nwin
window = np.hamming(nwin)
nn = range(nwin, len(x), nstep)

X = np.zeros((len(nn), nfft // 2))
for i, n in enumerate(nn):
    xseg = x[n-nwin:n]
    z = np.fft.fft(window * xseg, nfft)
    X[i, :] = np.log(np.abs(z[:nfft // 2]))

plt.imshow(X.T, interpolation='nearest', origin='lower', aspect='auto')
plt.show()

# %% [markdown]
# ## Feature Extraction

# %%
# Extract features (MFCCs) from the audio files and store them in a list
path = sample_audio_path
lst = []
start_time = time.time()

for subdir, dirs, files in os.walk(path):
    for file in files:
        try:
            X, sample_rate = librosa.load(os.path.join(subdir, file), res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            label = int(file[7:8]) - 1  # Convert labels from 1-8 to 0-7
            lst.append((mfccs, label))
        except ValueError:
            continue

print(f"--- Data loaded. Loading time: {time.time() - start_time:.2f} seconds ---")

# %%
X, y = zip(*lst)
X = np.asarray(X)
y = np.asarray(y)

# %%
save_dir = 'Saved_Models'
os.makedirs(save_dir, exist_ok=True)
joblib.dump(X, os.path.join(save_dir, 'X.joblib'))
joblib.dump(y, os.path.join(save_dir, 'y.joblib'))

# %%
X = joblib.load(os.path.join(save_dir, 'X.joblib'))
y = joblib.load(os.path.join(save_dir, 'y.joblib'))

# %% [markdown]
# ## Split the Data into Training and Testing Sets

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=11)

# %% [markdown]
# ## Model Training and Evaluation

# %% [markdown]
# ### Decision Tree Model

# %%
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print(classification_report(y_test, predictions, zero_division=0))

# %% [markdown]
# ### Random Forest Model

# %%
rforest = RandomForestClassifier(
    criterion="gini", max_depth=10, max_features="log2", max_leaf_nodes=100,
    min_samples_leaf=3, min_samples_split=20, n_estimators=22000, random_state=5
)
rforest.fit(X_train, y_train)
predictions = rforest.predict(X_test)
print(classification_report(y_test, predictions, zero_division=0))

# %% [markdown]
# ### Neural Network Model

# %%
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

model = Sequential([
    Conv1D(128, 5, padding='same', input_shape=(40, 1)),
    Activation('relu'),
    Dropout(0.1),
    MaxPooling1D(pool_size=8),
    Conv1D(128, 5, padding='same'),
    Activation('relu'),
    Dropout(0.1),
    Flatten(),
    Dense(8),
    Activation('softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
# Train the model
history = model.fit(X_train_cnn, y_train, batch_size=16, epochs=1000, validation_data=(X_test_cnn, y_test))

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# %%
predictions = np.argmax(model.predict(X_test_cnn), axis=1)
print(classification_report(y_test.astype(int), predictions, zero_division=0))
print(confusion_matrix(y_test.astype(int), predictions))

# %% [markdown]
# ## Save the Trained Model

# %%
model_name = 'Emotion_Voice_Detection_Model.keras'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print(f'Saved trained model at {model_path}')

# %%
loaded_model = tf.keras.models.load_model(model_path)
loaded_model.summary()
loss, acc = loaded_model.evaluate(X_test_cnn, y_test)
print(f"Restored model, accuracy: {acc * 100:.2f}%")


