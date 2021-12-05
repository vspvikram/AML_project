import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras import Sequential
import os.path
from PIL import Image

path = "/users/vspvikram/Downloads/AML_project/CheXNet-Keras-master/"
weight_file = "best_weights.h5"
input_shape = (None, 224, 224, 3)
classes = 14

class Encoder(tf.keras.Model):
    def __init__(self, classes=14, input_shape = (224, 224, 3)):
        super(Encoder, self).__init__()
        model = tf.keras.applications.densenet.DenseNet121(weights=os.path.join(path, weight_file),
                            input_shape=input_shape,
                            classes=classes)
        self.model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
#         self.model.build(input_shape = input_shape)
        
    def call(self, x):
        return self.model(x)

