from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Activation,Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import Sequential


model = Sequential()
input_shape = (img_size,img_size,channels)


model.add(Conv2D(6, (5, 5), input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(120))
model.add(Activation('relu'))


model.add(Dense(84))
model.add(Activation('relu'))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
