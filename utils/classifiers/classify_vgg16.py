import tensorflow as tf
import numpy as np
import cv2

image_width=112
image_height=112
def build_model(classes_num = 5):
    pretrained_model = tf.keras.applications.VGG16(input_shape = (image_height, image_width, 3), include_top = False, weights = 'imagenet')
    for layer in pretrained_model.layers[:-4]:
        layer.trainable = False

    # for layer in pretrained_model.layers:
    #     print(layer, layer.trainable)

    model = tf.keras.models.Sequential()

    model.add(pretrained_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation = 'relu'))
    model.add(tf.keras.layers.Dense(512, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(classes_num, activation = 'softmax'))
    return model

model = build_model()
model.load_weights('/storage/weights/vgg16/day_500.h5')

def predict_class(img):
    img = cv2.resize(img, (image_width, image_height))
    x = model.predict(img[np.newaxis, :])
    return np.argmax(x)