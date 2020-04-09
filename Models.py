import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

np.random.seed(12)
import tensorflow
tensorflow.set_random_seed(12)
import time




from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten
from keras import optimizers
from keras.models import Model, load_model
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from hyperas import optim
from hyperas.distributions import choice, uniform, loguniform
from hyperopt import Trials, STATUS_OK, tpe, hp
from keras.optimizers import RMSprop, Adadelta, Adagrad, Nadam, Adam
import global_config
from keras import callbacks
from sklearn.model_selection import train_test_split















def CNNPooling(params, input_shape, n_classes):

    X_input = Input(input_shape)
    X=Conv2D(32, (2, 2), activation=params['activation'], name = 'conv0')(X_input)
    X=MaxPooling2D(pool_size=(1,2), padding='same')(X)
    X=Dropout(params['dropout1'])(X)
    X = Conv2D(64, (2, 2), activation=params['activation'], name='conv1')(X)
    X=MaxPooling2D(pool_size=(2,2), padding='same')(X)
    X=Dropout(params['dropout2'])(X)
    X = Conv2D(128, (1, 2), activation=params['activation'], name='conv2')(X)
    X =Flatten()(X)
    X = Dense(256, activation='relu')(X)
    X = Dense(1024, activation='relu')(X)
    X = Dense(n_classes, activation='softmax')(X)
    model=Model(input=X_input, output=X)
    model.summary()
    model.compile(loss=params['losses'],
                        optimizer=params['optimizer'] (lr=params['lr']),
                        metrics=['acc'])
    return model


def CNN(params, input_shape, n_classes):

    X_input = Input(input_shape)
    X=Conv2D(32, (2, 2), activation=params['activation'], name = 'conv0',  kernel_initializer='glorot_uniform')(X_input)
    X=Dropout(params['dropout1'])(X)
    X = Conv2D(64, (2, 2), activation=params['activation'], name='conv1',  kernel_initializer='glorot_uniform')(X)
    X=Dropout(params['dropout2'])(X)
    X = Conv2D(128, (1, 2), activation=params['activation'], name='conv2',  kernel_initializer='glorot_uniform')(X)
    X =Flatten()(X)
    X = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(X)
    X = Dense(1024, activation='relu',  kernel_initializer='glorot_uniform')(X)
    X = Dense(n_classes, activation='softmax')(X)
    model=Model(input=X_input, output=X)
    model.summary()
    model.compile(loss=params['losses'],
                        optimizer=params['optimizer'] (lr=params['lr']),
                        metrics=['acc'])
    return model


def Autoencoder(x_train, y_train, x_test, y_test):
    input_shape = (x_train.shape[1],)
    input2 = Input(input_shape)


    # encoder_layer
    # Dropoout?
    #  input1 = Dropout(.2)(input)
    encoded = Dense(80, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod0')(input2)
    encoded = Dense(30, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod1')(encoded)
    encoded = Dense(10, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='encod2')(encoded)

    encoded= Dropout({{uniform(0, 1)}})(encoded)
    decoded = Dense(30, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='decoder1')(encoded)
    decoded = Dense(80, activation='relu',
                    kernel_initializer='glorot_uniform',
                    name='decoder2')(decoded)
    decoded = Dense(x_train.shape[1], activation='linear',
                    kernel_initializer='glorot_uniform',
                    name='decoder3')(decoded)


    model = Model(inputs=input2, outputs=decoded)
    model.summary()

    adam=Adam(lr={{uniform(0.0001, 0.01)}})
    model.compile(loss='mse', metrics=['acc'],
                  optimizer=adam)
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                restore_best_weights=True),
    ]
    XTraining, XValidation, YTraining, YValidation = train_test_split(x_train, y_train, stratify=y_train,
                                                                      test_size=0.2)  # before model building

    tic = time.time()
    history= model.fit(XTraining, YTraining,
                      batch_size={{choice([32,64, 128,256,512])}},
                      epochs=150,
                      verbose=1,
                      callbacks=callbacks_list,
                      validation_data=(XValidation,YValidation))

    toc = time.time()

    # get the highest validation accuracy of the training epochs
    score = np.amin(history.history['val_loss'])
    print('Best validation loss of epoch:', score)


    scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
    score = min(scores)
    print('Score',score)


    print('Best score',global_config.best_score)




    if global_config.best_score > score:
        global_config.best_score = score
        global_config.best_model = model
        global_config.best_numparameters = model.count_params()

        best_time = toc - tic



    return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(history.history['loss']), 'n_params': model.count_params(), 'model': global_config.best_model, 'time':toc - tic}

