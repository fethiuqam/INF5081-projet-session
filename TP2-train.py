#!/usr/bin/env python

import sys
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pour desactiver le gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# vérifier le nombre d'arguments
if len(sys.argv) != 2:
    print ('Veuillez spécifier le chemin du repertoire dataset')
    exit(0)

data_dir = sys.argv[1]
model_save = 'model.h5'

# redimensionnement des images à 128x128
img_size = 128

# augmentation de donnees
datagen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.2,  # 80% train_set , 20% val_set
                             fill_mode='nearest', 
                             shear_range=0.2,
                             horizontal_flip=True,
                             rotation_range=40,
                             brightness_range=[0.5,1.5],
                             width_shift_range=0.2,
                             height_shift_range=0.2)

# generation du train dataset
train_data = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(img_size, img_size),
    color_mode="rgb",
    class_mode='categorical',
    subset="training",
    batch_size = 256,
    shuffle=True,
    seed=42
)

# generation du validation dataset
test_data = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(img_size, img_size),
    color_mode="rgb",
    class_mode='categorical',
    subset="validation",
    batch_size = 1,
    shuffle=False,
    seed=42
)

# instanciation du model (architecture)
model = Sequential()

# 1ere couche conv pool relu : 24 filtres
model.add(Conv2D(filters=24, kernel_size=(3,3), input_shape=(img_size,img_size,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) # pour éviter l'overfitting

# 2eme couche conv pool relu : 128 filtres
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 3eme couche conv pool relu : 256 filtres
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 4eme couche conv pool relu : 768 filtres
model.add(Conv2D(filters=768, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# couche connectée FC
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))

# couche de sortie : fonction d'activation softmax
model.add(Dense(10,activation='softmax')) #10 équivaut au nombre de classes
model.summary()


# compilation du model :
# optimizer : adam
# fonction de cout : cross entropy

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# lancement de l'entrainement du model
# epochs : 800
# batch size : 128
history = model.fit(train_data, 
                    batch_size=128, 
                    epochs=800, 
                    verbose=1, 
                    validation_data=test_data)

# sauvegarde du model après l'entrainement
model.save(model_save)

# calcul du val_loss et val_accuracy
loss, acc = model.evaluate(test_data, verbose=0)
print("val loss = ", loss)
print("val accuracy = ", acc)


predict=model.predict(test_data)
predictions=np.argmax(predict,axis=1)

# affichage du rapport : precision moyenne / recall moyen
print(classification_report(test_data.classes,predictions)) 

# afficahge de la matrice de confusion
print(confusion_matrix(test_data.classes, predictions))

# affichage du graphe de la tendance : loss , accuracy
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) 
plt.show()