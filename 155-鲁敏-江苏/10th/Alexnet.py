import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization

#model
class Alexnet(tf.keras.Model):
    def __init__(self,num_classes=1000):
        super(Alexnet,self).__init__()

        self.features=tf.keras.Sequential([
            Conv2D(96,(11,11))
        ])