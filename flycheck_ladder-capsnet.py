import numpy as np

from keras import models
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Activation, Flatten
from keras.utils import to_categorical
from keras import backend as K

def ConvModel(input_shape, n_class):
    x = Input(shape=input_shape)
    conv1 = Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    conv2 = Conv2D(filters=32, kernel_size=5, activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=9, activation='relu')(conv2)
    dense1 = Flatten()(conv3)
    dense2 = Dense(128, activation='relu')(dense1)
    y = Dense(n_class, activation='softmax')(dense2)

    model = models.Model(x, y)
    return model

def train(model, data):
    (x_train, y_train), (x_test, y_test) = data

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=12,
              verbose=1, validation_data=(x_test, y_test))

def test(model, data):
    x_test, y_test = data
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def load_mnist():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

# ---------- Main Routing ---------- #
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = ConvModel(input_shape=x_train.shape[1:],
                      n_class=len(np.unique(np.argmax(y_train, 1))))
    model.summary()
    train(model=model, data=((x_train, y_train), (x_test, y_test)))
    test(model=model, data=((x_test, y_test)))
