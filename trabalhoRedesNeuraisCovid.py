import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras

import cv2
import os
from keras.datasets import mnist #Lecun, 1998

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D
# from keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers import Dropout, Input, MaxPooling2D

from keras import backend as K

import glob

import random

# ----------------------------------------------------------

teste = glob.glob("C:/Users/Nique/Desktop/COVID-19 Radiography Database/Teste/*")
treino = glob.glob("C:/Users/Nique/Desktop/COVID-19 Radiography Database/Treino/*")

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
    # print(x)
    imgAux = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    imgTreinoResized= cv2.resize(imgAux,dim, interpolation = cv2.INTER_AREA)
    imgTreino.append(imgTreinoResized)
    if "NORMAL" in x:
        valTreino.append(0)
    else:
        valTreino.append(1)
    count = count+1   


# randomização controlada das imagens
SEED = 10

c = list(zip(imgTeste, valTeste))
random.seed(SEED)
random.shuffle(c)
imgTeste, valTeste = zip(*c)

c = list(zip(imgTeste, valTeste))
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


# ainda n sei
imgTreino = imgTreino.reshape(imgTreino.shape[0],1,224,224).astype('float32')
imgTeste = imgTeste.reshape(imgTeste.shape[0],1,224,224).astype('float32')


# normalização dos valores dos pixels
imgTreino /= 255
imgTeste /= 255



# categorização dos possíveis outputs
valTeste = keras.utils.to_categorical(valTeste,2)
valTreino = keras.utils.to_categorical(valTreino,2)
print(valTeste.shape)


imgTreino = np.reshape(imgTreino,(len(imgTreino),224,224,1))
imgTeste = np.reshape(imgTeste,(len(imgTeste),224,224,1))

# construção da rede
model = Sequential()
model.add(Conv2D(30,(5,5), input_shape=(224,224,1), strides=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dense(2,activation='softmax', name='predict'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(imgTreino,valTreino,epochs=30, batch_size=16, validation_data=(imgTeste,valTeste))