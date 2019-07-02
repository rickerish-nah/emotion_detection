from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from parameters_keras import NETWORK, HYPERPARAMS




def build_model():

    if NETWORK.model == 'H':
        return build_modelH()
    else:
        print( "ERROR: no model " + str(NETWORK.model))
        exit()

def build_modelH():

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation=NETWORK.activation, input_shape=(NETWORK.input_size,NETWORK.input_size,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation=NETWORK.activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(HYPERPARAMS.dropout))

    model.add(Conv2D(128, kernel_size=(3, 3), activation=NETWORK.activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation=NETWORK.activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(HYPERPARAMS.dropout))

    model.add(Flatten())
    model.add(Dense(1024, activation=NETWORK.activation))
    model.add(Dropout(0.5))
    model.add(Dense(NETWORK.output_size, activation='softmax'))
    
    if HYPERPARAMS.optimizer == 'adam':
        optimizer = Adam(lr = HYPERPARAMS.learning_rate, decay = HYPERPARAMS.learning_rate_decay) #beta1=HYPERPARAMS.optimizer_param
    
    model.compile(loss=NETWORK.loss,optimizer=optimizer,metrics=['accuracy'])

    return model

