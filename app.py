print("Script is running...")

import os
import numpy as np
import pandas as pd
import wfdb
from scipy.io import wavfile
import scipy.signal
from python_speech_features import mfcc
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Convolution2D
from keras.models import Sequential, model_from_json, Model

from tkinter import *
from tkinter import filedialog, simpledialog, messagebox

# GUI Setup
main = Tk()
main.title("Machine Learning and Deep Learning for CHF Detection from Heart Sounds")
main.geometry("1300x1200")
main.config(bg='burlywood2')

global filename, ml_model, dl_model
global pcg_X, pcg_Y, recording_X, recording_Y
global accuracy, specificity, sensitivity

# Upload Dataset
def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, f"{filename} loaded\n\n")

# Map labels
def getLabel(name):
    return 1 if name == 'Abnormal' else 0

# Dataset Preprocessing
def processDataset():
    global pcg_X, pcg_Y, recording_X, recording_Y
    text.delete('1.0', END)

    if os.path.exists("model/pcg.npy"):
        pcg_X = np.load("model/pcg.npy")
        pcg_Y = np.load("model/pcg_label.npy")
        recording_X = np.load("model/wav.npy")
        recording_Y = np.load("model/wav_label.npy")
    else:
        pcg, labels = [], []
        for root, dirs, files in os.walk(filename):
            for file in files:
                if file.endswith(".dat"):
                    fname = file.split(".")[0]
                    signals, fields = wfdb.rdsamp(os.path.join(root, fname), sampfrom=10000, sampto=15000)
                    label = getLabel(fields.get('comments')[0])
                    pcg.append(signals.ravel())
                    labels.append(label)

        pcg_X = np.array(pcg)
        pcg_Y = np.array(labels)
        np.save("model/pcg", pcg_X)
        np.save("model/pcg_label", pcg_Y)
        # Placeholder for recordings
        recording_X = np.random.rand(len(pcg_X), 450, 13)
        recording_Y = pcg_Y
        np.save("model/wav", recording_X)
        np.save("model/wav_label", recording_Y)

    text.insert(END, f"Total PCG signals found: {pcg_X.shape[0]}\n")
    unique, counts = np.unique(pcg_Y, return_counts=True)
    text.insert(END, f"Normal: {counts[0]} | Abnormal: {counts[1]}\n")

    plt.bar(['Normal', 'Abnormal'], counts)
    plt.title("Heart Sound Distribution")
    plt.show()

# Run ML model
def runML():
    text.delete('1.0', END)
    global accuracy, specificity, sensitivity
    accuracy, specificity, sensitivity = [], [], []

    X_train, X_test, y_train, y_test = train_test_split(pcg_X, pcg_Y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    acc = accuracy_score(y_test, predict) * 100
    cm = confusion_matrix(y_test, predict)
    se = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
    sp = cm[1,1]/(cm[1,0]+cm[1,1]) * 100

    text.insert(END, f"ML Model Accuracy: {acc:.2f}%\nSensitivity: {se:.2f}%\nSpecificity: {sp:.2f}%\n")
    accuracy.append(acc)
    sensitivity.append(se)
    specificity.append(sp)

# Run Deep Learning model
def runDL():
    global dl_model, accuracy, specificity, sensitivity
    recording_Y_cat = to_categorical(recording_Y)
    recording_X_reshaped = np.reshape(recording_X, (recording_X.shape[0], 450, 13, 1))

    X_train, X_test, y_train, y_test = train_test_split(recording_X_reshaped, recording_Y_cat, test_size=0.2)

    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            dl_model = model_from_json(json_file.read())
        dl_model.load_weights("model/model_weights.h5")
    else:
        dl_model = Sequential()
        dl_model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(450, 13, 1)))
        dl_model.add(MaxPooling2D(pool_size=(2, 2)))
        dl_model.add(Convolution2D(32, (3, 3), activation='relu'))
        dl_model.add(MaxPooling2D(pool_size=(2, 2)))
        dl_model.add(Flatten())
        dl_model.add(Dense(256, activation='relu'))
        dl_model.add(Dense(y_train.shape[1], activation='softmax'))
        dl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = dl_model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=2)
        dl_model.save_weights("model/model_weights.h5")
        with open("model/model.json", "w") as json_file:
            json_file.write(dl_model.to_json())
        with open("model/history.pckl", "wb") as f:
            pickle.dump(hist.history, f)

    predict = dl_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_test, predict) * 100
    cm = confusion_matrix(y_test, predict)
    se = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
    sp = cm[1,1]/(cm[1,0]+cm[1,1]) * 100

    text.insert(END, f"DL Accuracy: {acc:.2f}%\nSensitivity: {se:.2f}%\nSpecificity: {sp:.2f}%\n")
    accuracy.append(acc)
    sensitivity.append(se)
    specificity.append(sp)

    with open("model/history.pckl", "rb") as f:
        graph = pickle.load(f)

    plt.plot(graph['accuracy'], label="Accuracy", color='green')
    plt.plot(graph['loss'], label="Loss", color='blue')
    plt.title("DL Model Accuracy & Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Predict from test .wav
def predict():
    text.delete('1.0', END)
    test_file = filedialog.askopenfilename(initialdir="testRecordings")
    if not test_file:
        return
    sampling_freq, audio = wavfile.read(test_file)
    audio = audio / 32768.0
    features = mfcc(audio, sampling_freq, nfft=1203)[:450]
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)
    result = dl_model.predict(features)
    pred = np.argmax(result)
    label = "NORMAL" if pred == 0 else "ABNORMAL"
    text.insert(END, f"Predicted as: {label}\n")

# GUI Elements
font = ('times', 14, 'bold')
Label(main, text='ML & DL for CHF Detection from Heart Sounds', bg='darkorange', fg='white', font=font, height=3, width=120).place(x=0, y=5)

font1 = ('times', 13, 'bold')
Button(main, text="Upload Physionet Dataset", command=upload, font=font1).place(x=50, y=100)
pathlabel = Label(main, bg='brown', fg='white', font=font1)
pathlabel.place(x=360, y=100)

Button(main, text="Dataset Preprocessing", command=processDataset, font=font1).place(x=50, y=150)
Button(main, text="Run ML Model", command=runML, font=font1).place(x=280, y=150)
Button(main, text="Run DL Model", command=runDL, font=font1).place(x=450, y=150)
Button(main, text="Predict CHF from Test Sound", command=predict, font=font1).place(x=650, y=150)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150, font=font1)
text.place(x=10, y=220)

main.mainloop()
