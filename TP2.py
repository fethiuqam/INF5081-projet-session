#!/usr/bin/env python

import sys
from tensorflow import keras
import numpy as np
from keras.preprocessing import image

# pour desactiver le gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# ne pas afficher tensorflow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# vérifier le nombre d'arguments
if len(sys.argv) != 2:
    print ('Veuillez spécifier le chemin de l\'image à prédir en argument')
    exit(0)

# caracteristiques du model 
img_size = 128
model_h5 = 'model.h5'

# etiquettes des classes
especes = ['Armillaria lutea', 'Coprinellus micaceus','Fomes formentarius','Fomitopsis pinicola', 'Ganoderma pfeifferi', 'Mycena galericulata', 'Pluteus cervinus', 'Plicatura crispa', 'Tricholoma scalpturatum', 'Xerocomellus chrysentron']
comestibles = [False, False, False, False, True, True, True, False, True, True]

image_file = sys.argv[1]

# chargment du model entrainé
model = keras.models.load_model(model_h5)

test_image = image.load_img(image_file, target_size=(img_size,img_size))

# prédiction de la classe de l'image
result = model.predict(np.expand_dims(image.img_to_array(test_image), axis=0))

index = result.argmax(axis=-1)

# calcul de la probabilité de comstibilité de la prédiction
comestible = 0
for i in range(10):
    if(comestibles[i]):
        comestible += result[0][i]

# affichage des resultats
print ('Espece : ', especes[index[0]])        
print ('Comestible : ', "Oui" if comestible > 0.50 else 'Non')
