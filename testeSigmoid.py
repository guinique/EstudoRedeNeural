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

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))

import glob

import random

# ----------------------------------------------------------
teste = glob.glob("COVID-19_Radiography_Dataset/COVID/*")
treino = glob.glob("COVID-19_Radiography_Dataset/Normal/*")

# leitura das imagens e dos valores de teste
dim = (224, 224)
imgTeste = []
valTeste = []
count = 0
for x in teste:
    imgAux = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    imgTesteResized= cv2.resize(imgAux,dim, interpolation = cv2.INTER_AREA)
    imgTeste.append(imgTesteResized)
    if "NORMAL" in x:
        valTeste.append(0)
    else:
        valTeste.append(1)
    count = count+1

# leitura das imagens e dos valores de treino
imgTreino = []
valTreino = []
count = 0
for x in treino:
    imgAux = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    imgTreinoResized= cv2.resize(imgAux,dim, interpolation = cv2.INTER_AREA)
    imgTreino.append(imgTreinoResized)
    if "NORMAL" in x:
        valTreino.append(0)
    else:
        valTreino.append(1)
    count = count+1   

# randomização controlada das imagens
SEED = 93

c = list(zip(imgTeste, valTeste))
random.seed(SEED)
random.shuffle(c)
imgTeste, valTeste = zip(*c)

c = []

c = list(zip(imgTreino, valTreino))
random.seed(SEED)
random.shuffle(c)
imgTreino, valTreino = zip(*c)

# teste para saber se a randomização ocorreu corretamente
# print(valTeste[0])
# plt.imshow(imgTeste[0])
# plt.show()

# conversão de lista para array
imgTreino = np.array(imgTreino)
imgTeste = np.array(imgTeste)


# conversão para o shape correto
imgTreino = imgTreino.reshape(imgTreino.shape[0],1,224,224).astype('float32')
imgTeste = imgTeste.reshape(imgTeste.shape[0],1,224,224).astype('float32')


# normalização dos valores dos pixels
imgTreino /= 255
imgTeste /= 255


# Conversão dos valores em tupla para array
valTreino = np.array(valTreino)
valTeste = np.array(valTeste)

valTreino = np.reshape(valTreino,(len(valTreino),1))
valTeste = np.reshape(valTeste,(len(valTeste),1))

imgTreino = np.reshape(imgTreino,(len(imgTreino),224,224,1))
imgTeste = np.reshape(imgTeste,(len(imgTeste),224,224,1))

batch_size = 32

# construção da rede
model = Sequential()
model.add(Conv2D(6,(3,3), input_shape=(224,224,1), strides=(1,1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(1,activation='sigmoid', name='predict'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# treinamento do modelo
model.fit(imgTreino,valTreino,epochs=10, batch_size=batch_size, validation_data=(imgTeste,valTeste))

# leitura dos dados de validação
validacao = glob.glob("C:/Users/Nique/Desktop/COVID-19 Radiography Database/Validacao/*")
imgValid = []
valValid = []
count = 0
for x in validacao:
    imgAux = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    imgValidResized= cv2.resize(imgAux,dim, interpolation = cv2.INTER_AREA)
    imgValid.append(imgValidResized)
    if "NORMAL" in x:
        valValid.append(0)
    else:
        valValid.append(1)
    count = count+1   

imgValid = np.array(imgValid)
imgValid = np.reshape(imgValid,(len(imgValid),224,224,1))

print(valValid[0])
print(valValid[1])
print(valValid[2])
print(valValid[3])
print(valValid[4])
print(valValid[5])

predictions = model.predict_classes(imgValid, batch_size=batch_size, verbose=0)

# print(predictions[0])
# print(predictions[1])
# print(predictions[2])
# print(predictions[3])
# print(predictions[4])
# print(predictions[5])




matriz_confusao = confusion_matrix(valValid, predictions)


#função para plot da matriz de confusão https://deeplizard.com/learn/video/km7pxKy4UHU
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


cm_plot_labels = ['Covid -','Covid +']
plot_confusion_matrix(matriz_confusao, cm_plot_labels, title='Confusion Matrix')