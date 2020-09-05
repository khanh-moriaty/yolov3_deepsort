import os

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import numpy as np
import cv2


json_file = open('/storage/mobile_net/MobileNets_traffic_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)



labels = LabelEncoder()
labels.classes_ = np.load('/storage/mobile_net/traffic_classes.npy')
print("[INFO] Labels loaded successfully...")
y = labels.classes_

def create_model(lr=1e-4,decay=1e-4/25, training=False,output_shape=12):
    baseModel = MobileNetV2(weights="imagenet", 
                            include_top=True,
                            input_tensor=Input(shape=(80, 80, 3)))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(output_shape, activation="softmax")(headModel)
    
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    if training:
        # define trainable lalyer
        for layer in baseModel.layers:
            layer.trainable = True
        # compile model
        optimizer = Adam(lr=lr, decay = decay)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=["accuracy"])    
        
    return model

# model = create_model()
model.load_weights("/storage/mobile_net/traffic_recognition.h5")

print("[INFO] Model loaded successfully...")

class_mapping = [0,1,1,-1,-1,2,2,2,-1,3,3,4,4,4]

def predict_class(image):
    image = cv2.resize(image,(80,80))
    # image = np.stack((image,)*3, axis=-1)
    feature = model.predict(image[np.newaxis,:])
    max_arr = np.argmax(feature)
    prediction = labels.inverse_transform([max_arr])
    prediction = class_mapping[int(prediction)]
    return prediction