# -*- coding: utf-8 -*-

import itertools
import numpy as np
from matplotlib import pyplot as plt
# from keras.utils import np_utils
import tensorflow.keras as keras

import cv2
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D
from tensorflow.python.keras.layers import Dropout, Input, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import backend as K
from tensorflow.keras import callbacks

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.utils.multiclass import type_of_target
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))

import glob

import random

# ----------------------------------------------------------
covid = glob.glob("COVID-19_Radiography_Dataset/COVID/*")
normal = glob.glob("COVID-19_Radiography_Dataset/Normal/*")
pneumonia = glob.glob("COVID-19_Radiography_Dataset/Viral Pneumonia/*")

# Directory
directory = "pastaNaoNomeada"
  
# Parent Directory path
parent_dir = "C:/Users/Nique/Desktop/resultados_rede"
  
# Path
path = os.path.join(parent_dir, directory)

os.mkdir(path)

# nome = []
# leitura das imagens e dos valores de teste
dim = (299, 299)
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


imgTeste, imgValid, valTeste, valValid = train_test_split(imgs,vals, test_size=0.3, random_state=42)


x_train, x_test, y_train, y_test = train_test_split(imgTeste,valTeste, test_size=0.3, random_state=42)

x_train = np.array(x_train)
x_test = np.array(x_test)
imgValid = np.array(imgValid)

x_train = np.reshape(x_train,(len(x_train),224,224,1)).astype('float32')
x_test = np.reshape(x_test,(len(x_test),224,224,1)).astype('float32')
imgValid = np.reshape(imgValid,(len(imgValid),224,224,1)).astype('float32')

# normalização dos valores dos pixels
x_test /= 255
x_train /= 255
imgValid /= 255

y_train = np.array(y_train)
y_test = np.array(y_test)
valValid = np.array(valValid)

lalala_train = keras.utils.to_categorical(y_train,3)
lalala_test = keras.utils.to_categorical(y_test,3)

inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

np.random.seed(42)

batch_size = 290
epochs = 50
cm_array = []
# K-fold Cross Validation model evaluation
opt = keras.optimizers.Adam(learning_rate=0.0003)

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 3, 
                                        restore_best_weights = True)
kfold = StratifiedKFold(n_splits=5, shuffle=True)
# print(y_train[0])
# print(y_train[1])
# print(y_train[2])
# print(y_train[3])
# print(y_train[4])
# print('----------------------')
# print(y_test[0])
# print(y_test[1])
# print(y_test[2])
# print(y_test[3])
# print(y_test[4])

val_loss = []
val_accuracy = []


tp = []
tn = []
fp = []
fn = []

# precision
pre = []

# recall
rec = []

# Accuracy
acc = []

# Specificity
spe = []




for train, test in kfold.split(inputs, targets):

    # unique, counts = np.unique(targets[train], return_counts=True)
    # print(dict(zip(unique, counts)))

    # unique, counts = np.unique(targets[test], return_counts=True)
    # print(dict(zip(unique, counts)))

    # plt.imshow(inputs[train][0])
    # plt.show()
    # plt.imshow(inputs[test][0])
    # plt.show()

    weights=class_weight.compute_class_weight('balanced', np.unique(targets[train]), targets[train])
    weights = {l:c for l,c in zip(np.unique(targets[train]), weights)}
    print(weights)

    print('----------------------------------------------------------------')
    print(weights)

    auxTrain = keras.utils.to_categorical(targets[train],3)
    auxTest = keras.utils.to_categorical(targets[test],3)

    
    # FAZER SISTEMA DOS WEIGTHS DAS CLASSES
    # FAZER SISTEMA DOS WEIGTHS DAS CLASSES
    # FAZER SISTEMA DOS WEIGTHS DAS CLASSES
    # FAZER SISTEMA DOS WEIGTHS DAS CLASSES
    # FAZER SISTEMA DOS WEIGTHS DAS CLASSES
    # FAZER SISTEMA DOS WEIGTHS DAS CLASSES
    # FAZER SISTEMA DOS WEIGTHS DAS CLASSES
    # weights=class_weight.compute_class_weight('balanced', np.unique(targets[train]), targets[train])

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
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    # treinamento do modelo
    # history = model.fit(inputs[train], targets[train],epochs=epochs, batch_size=batch_size, validation_data=(inputs[test], targets[test]), callbacks=[earlystopping])  
    history = model.fit(inputs[train], auxTrain,epochs=epochs, batch_size=batch_size, validation_data=(inputs[test], auxTest), class_weight=weights)  

    print('loss e accuracy')
    print(history.history['val_loss'])
    print(history.history['val_accuracy'])
    val_loss.append(history.history['val_loss'])
    val_accuracy.append(history.history['val_accuracy'])

    pred = np.argmax(model.predict(imgValid), axis=1)

    matriz_confusao = confusion_matrix(valValid, pred, labels=[0,1,2])

    # print(matriz_confusao)

    cm_array.append(matriz_confusao)
    # print(cm_array)
    tp = matriz_confusao[0][0]

    tn = matriz_confusao[1][1]
    tn += matriz_confusao[1][2]
    tn += matriz_confusao[2][1]
    tn += matriz_confusao[2][2]

    fp = matriz_confusao[0][1]
    fp += matriz_confusao[0][2]

    fn = matriz_confusao[1][0]
    fn += matriz_confusao[2][0]


    pre.append(tp/(tp+fp))
    rec.append(tp/(tp+fn))
    spe.append(tn/(tn+fp))
    acc.append((tp+tn)/(tp+tn+fp+fn))


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
        plt.savefig(path+"/media_matrizes.png")
        plt.show()


# print("metricas - valores de cada fold e desvio padrão final")
# print("\nprecision")
# print(pre)
# print(np.std(pre))
# print("\nspecificity")
# print(spe)
# print(np.std(spe))
# print("\naccuracy")
# print(acc)
# print(np.std(acc))
# print("\nrecall")
# print(rec)
# print(np.std(rec))
f= open(path+"/calcs.txt","w+")
f.write('precision \n')
f.write(str(pre)+'\n')
f.write(str(np.std(pre))+"\n\n")
f.write('specificity \n')
f.write(str(spe)+'\n')
f.write(str(np.std(spe))+"\n\n")
f.write('accuracy \n')
f.write(str(acc)+'\n')
f.write(str(np.std(acc))+"\n\n")
f.write('recall \n')
f.write(str(rec)+'\n')
f.write(str(np.std(rec))+"\n\n")
f.close()

cm_plot_labels = ['Covid -','Covid +','Pneumonia Viral']
print(np.mean(cm_array, axis=0))
mediakfold = np.floor(np.mean(cm_array, axis=0))

plot_confusion_matrix(mediakfold, cm_plot_labels, title='Confusion Matrix')

for x in range(len(val_accuracy)):

    plt.title("val_accuracy - fold "+str(x)+"")
    plt.plot(val_accuracy[x])
    plt.ylim([0,1])
    plt.savefig(path+'/val_accuracy_'+str(x)+'.png')
    plt.show()

    plt.title("val_loss - fold "+str(x)+"")
    plt.plot(val_loss[x])
    plt.ylim([0,1])
    plt.savefig(path+'/val_loss_'+str(x)+'.png')
    plt.show()