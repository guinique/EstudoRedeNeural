# -*- coding: utf-8 -*-

import itertools
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras

import cv2
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D
from tensorflow.python.keras.layers import Dropout, Input, MaxPooling2D, AveragePooling2D

from keras import backend as K
from keras import callbacks

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))

import glob

import random

# ----------------------------------------------------------
covid = glob.glob("COVID-19_Radiography_Dataset/COVID/*")
normal = glob.glob("COVID-19_Radiography_Dataset/Normal/*")
pneumonia = glob.glob("COVID-19_Radiography_Dataset/Viral Pneumonia/*")
# nome = []
# leitura das imagens e dos valores de teste
dim = (224, 224)
imgCovid = []
valCovid = []
for x in covid:
    # nome.append(x)
    imgAux = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    imgCovidResized= cv2.resize(imgAux,dim, interpolation = cv2.INTER_AREA)
    imgCovid.append(imgCovidResized)
    valCovid.append(1)

imgNormal = []
valNormal = []
for x in normal:
    # nome.append(x)
    imgAux = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    imgNormalResized= cv2.resize(imgAux,dim, interpolation = cv2.INTER_AREA)
    imgNormal.append(imgNormalResized)
    valNormal.append(0)


imgPneumonia = []
valPneumonia = []
for x in pneumonia:
    # nome.append(x)
    imgAux = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    imgPneumoniaResized= cv2.resize(imgAux,dim, interpolation = cv2.INTER_AREA)
    imgPneumonia.append(imgPneumoniaResized)
    valPneumonia.append(2)

imgs = imgCovid + imgNormal + imgPneumonia
vals = valCovid + valNormal + valPneumonia


# # randomização controlada das imagens
# SEED = 10

# # c = list(zip(imgs, vals, nome))
# c = list(zip(imgs, vals))
# random.seed(SEED)
# random.shuffle(c)
# # imgs, vals, nome = zip(*c)
# imgs, vals = zip(*c)

# # print(nome[0])
# print(vals[0])
# plt.imshow(imgs[0], cmap="gray")
# plt.show()

# conversão de lista para array
# imgValid = np.array(imgValid)
# imgTeste = np.array(imgTeste)

imgTeste, imgValid, valTeste, valValid = train_test_split(imgs,vals, test_size=0.3, random_state=42)


x_train, x_test, y_train, y_test = train_test_split(imgTeste,valTeste, test_size=0.3, random_state=42)


print("y")
print(y_test[0])
print(y_test[1])
print(y_test[2])
print(y_test[3])
print(y_test[4])
print(y_test[5])
print(y_test[6])
print(y_test[7])

print("img")
print(valTeste[0])
print(valTeste[1])
print(valTeste[2])
print(valTeste[3])
print(valTeste[4])
print(valTeste[5])
print(valTeste[6])
print(valTeste[7])

x_train = np.array(x_train)
x_test = np.array(x_test)
imgValid = np.array(imgValid)

y_train = np.array(y_train)
y_test = np.array(y_test)
valValid = np.array(valValid)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
imgValid = imgValid.astype('float32')

# normalização dos valores dos pixels
x_test /= 255
x_train /= 255
imgValid /= 255

# print("aaaaaaaaaaaaaaaaaaaaaaa",y_train[0])

y_train = keras.utils.to_categorical(y_train,3)
y_test = keras.utils.to_categorical(y_test,3)
# valValid = keras.utils.to_categorical(y_test,3)

# print("aaaaaaaaaaaaaaaaaaaaaaa",y_train[0])
x_train = np.reshape(x_train,(len(x_train),224,224,1))
x_test = np.reshape(x_test,(len(x_test),224,224,1))
imgValid = np.reshape(imgValid,(len(imgValid),224,224,1))

inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)


batch_size = 32
epochs = 10
cm_array = []
# K-fold Cross Validation model evaluation
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)
kfold = KFold(n_splits=5, shuffle=True)
for train, test in kfold.split(inputs, targets):

    # construção da rede
    model = Sequential()
    model.add(Conv2D(6,(3,3), input_shape=(224,224,1), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16,(3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120,activation='relu'))
    model.add(Dense(84,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(3,activation='softmax', name='predict'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # treinamento do modelo
    model.fit(inputs[train], targets[train],epochs=epochs, batch_size=batch_size, validation_data=(inputs[test], targets[test]))  

    pred = np.argmax(model.predict(imgValid), axis=1)

    matriz_confusao = confusion_matrix(valValid, pred, labels=[0,1,2])

    # print(matriz_confusao)

    cm_array.append(matriz_confusao)
    print(cm_array)

    # #função para plot da matriz de confusão https://deeplizard.com/learn/video/km7pxKy4UHU
    def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


# print(cm_array)
cm_plot_labels = ['Covid -','Covid +','Pneumonia Viral']
print(np.mean(cm_array, axis=0))
mediakfold = np.floor(np.mean(cm_array, axis=0))

plot_confusion_matrix(mediakfold, cm_plot_labels, title='Confusion Matrix')