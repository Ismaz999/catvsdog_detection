import numpy as np #algèbre linéaire
import pandas as pd  # data processing
import cv2
import matplotlib.pyplot as plt #bibliothèque matplotlib
from tensorflow import keras
from keras.models import Sequential #permet de créer un empilement de couche (entrée et sortie)
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D 
import os
import zipfile

#with zipfile.ZipFile("/home/audensiel/projet/test1.zip","r") as z:
   # z.extractall(".")

#with zipfile.ZipFile("/home/audensiel/projet/train.zip","r") as z:
    #z.extractall(".")

main_dir = "/home/audensiel/projet/"
train_dir = "train"
path = os.path.join(main_dir, train_dir)

# ==== DATA PROCESSING====

X = []
y = []
convert = lambda category : int(category == 'dog')
def create_test_data(path):
    for p in os.listdir(path):
        category = p.split(".")[0]
        category = convert(category)
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append(category)

create_test_data(path)
X = np.array(X).reshape(-1, 80,80,1)
y = np.array(y)

#Normaliser les valeurs de gris en 0 et 1 plutot que 0 et 255
X = X/255.0

model = Sequential()

model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size = (2,2)))
# Add another:
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

test_dir = "test1"
path = os.path.join(main_dir,train_dir)
#os.listdir(path)

X_test = []
id_line = []
def create_test1_data(path):
    for p in os.listdir(path):
        id_line.append(p.split(".")[0])
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X_test.append(new_img_array)
        
create_test1_data(path)
X_test = np.array(X_test).reshape(-1,80,80,1)
X_test = X_test/255

predictions = model.predict(X_test)

predicted_val = [int(round(p[0])) for p in predictions]


submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})


submission_df.to_csv("submission.csv", index=False)

