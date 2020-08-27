import tensorflow as tf
import numpy as np
import cv2

image_width=112
image_height=112
def build_model(classes_num = 5):
    pretrained_model = tf.keras.applications.ResNet50V2(input_shape = (image_height, image_height, 3), include_top = False, weights = 'imagenet')
    for layer in pretrained_model.layers[:-15]:
        layer.trainable = False

    model = tf.keras.models.Sequential()

    model.add(pretrained_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation = 'relu'))
    model.add(tf.keras.layers.Dense(512, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(classes_num, activation = 'softmax'))
    return model

model = build_model()
model.load_weights('/storage/weights/classifier/resnet50v2/resnet50v2_day.h5')

def predict_class(img):
    img = cv2.resize(img, (image_width, image_height))
    x = model.predict(img[np.newaxis, :])
    return np.argmax(x)