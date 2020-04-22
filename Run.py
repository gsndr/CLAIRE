import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

np.random.seed(12)
import tensorflow
import itertools
tensorflow.random.set_seed(12)
import Plot
import Preprocessing as prp
import Models

from keras import optimizers
from keras import callbacks
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from keras.models import Model
from keras.models import load_model
import time
from keras import backend as K

np.set_printoptions(suppress=True)
from Clustering import clusteringMiniBatchKMeans
from Utils import reshapeFeature, saveNpArray, getResult
from sklearn.metrics import pairwise_distances_argmin_min

import AutoencoderHypersearch as ah
import EncodingOrding as eo
from DatasetConfig import Datasets
from sklearn.model_selection import train_test_split


class Execution():
    def __init__(self, dsConfig, config):
        self.config = config
        self.ds = dsConfig
        self.CLUSTERS = int(self.config.get(
            'CLUSTERS'))  # Se 1 le immagini vengono create tramite cluster, se 0 le immagini vengono create considerando gli esempi del training
        self.TRAIN = int(self.config.get('TRAIN'))  # Se 1 vengono create immagini del training set
        self.TEST = int(self.config.get('TEST'))  # Se 1 vengono create immagini del testing set
        self.num_cluster = int(self.config.get('NUM_CLUSTERS'))
        self.LOAD_CLUSTERS = int(self.config.get('LOAD_CLUSTERS'))
        self.nearest = self.config.get('NEAREST')
        fileOutput = self.ds.get('pathTime') + str(self.num_cluster) + 'result' + self.ds.get('testPath') + '.txt'
        self.file = open(fileOutput, 'w')
        self.file.write('Result time for: %s clusters \n' % self.num_cluster)
        self.file.write('\n')



    def distance(self, X_Train, X_Test, dfNormal, dfAttack, nearest):
        pathCluster = self.ds.get('pathDatasetCluster')
        X_Normal = np.array(dfNormal.drop(['classification'], 1).astype(float))
        X_Attack = np.array(dfAttack.drop(['classification'], 1).astype(float))

        row_attack = np.size(X_Attack, 0)
        row_normal = np.size(X_Normal, 0)

        if self.CLUSTERS == 1:
            tic = time.time()  # recupera il tempo corrente in secondi
            if (self.LOAD_CLUSTERS == 0):
                centerAtt = clusteringMiniBatchKMeans(X_Attack, self.num_cluster, row_attack)
                print("AttDone")
                np.save(pathCluster + "Centroid/centerAtt" + str(self.num_cluster) + ".npy", centerAtt)
                centerNorm = clusteringMiniBatchKMeans(X_Normal, self.num_cluster, row_normal)
                np.save(pathCluster + "Centroid/centerNorm" + str(self.num_cluster) + ".npy", centerNorm)
                print("NormDone")
                toc = time.time()
                self.file.write("Time Creation Clusters: %s" % (toc - tic))
                self.file.write('\n')
            else:
                centerAtt = np.load(pathCluster + "Centroid/centerAtt" + str(self.num_cluster) + ".npy")
                centerNorm = np.load(pathCluster + "Centroid/centerNorm" + str(self.num_cluster) + ".npy")
        else:

            centerAtt = X_Attack
            centerNorm = X_Normal

        matriceDistTrain = []
        matriceDistTest = []

        if self.TRAIN == 1:
            tic = time.time()
            for i in range(len(X_Train)):
                feature1 = reshapeFeature(X_Train[i])
                dist_matrixN = pairwise_distances_argmin_min(feature1, centerNorm)
                if nearest == True:
                    if dist_matrixN[1] == 0:
                        ind = dist_matrixN[0]
                        centerNorm[ind, :] = 0
                        dist_matrixN = pairwise_distances_argmin_min(feature1, centerNorm)
                        centerNorm[ind] = feature1

                dist_matrixA = pairwise_distances_argmin_min(feature1, centerAtt)
                if nearest == True:
                    if dist_matrixA[1] == 0:
                        ind = dist_matrixA[0]
                        centerAtt[ind, :] = 0
                        dist_matrixA = pairwise_distances_argmin_min(feature1, centerAtt)
                        centerAtt[ind] = feature1

                # row = [X_Train[i], centerNorm[dist_matrixN[0].item()], centerAtt[dist_matrixA[0].item()]]
                row = [centerNorm[dist_matrixN[0].item()], X_Train[i], centerAtt[dist_matrixA[0].item()]]
                matriceDistTrain.append(row)
            toc = time.time()
            self.file.write("Time Creation Training Images  : %s " % (toc - tic))
            self.file.write('\n')

        if self.TEST == 1:
            tic = time.time()
            for i in range(len(X_Test)):
                feature1 = reshapeFeature(X_Test[i])
                dist_matrixN = pairwise_distances_argmin_min(feature1, centerNorm)
                dist_matrixA = pairwise_distances_argmin_min(feature1, centerAtt)
                # row = [X_Test[i], centerNorm[dist_matrixN[0].item()], centerAtt[dist_matrixA[0].item()]]
                row = [centerNorm[dist_matrixN[0].item()], X_Test[i], centerAtt[dist_matrixA[0].item()]]
                matriceDistTest.append(row)
            toc = time.time()
            self.file.write("Time Creation Testing Images :  %s " % (toc - tic))
            self.file.write('\n')

        return matriceDistTrain, matriceDistTest



    def run(self):



        dsConf = self.ds
        print(dsConf.get('testPath') + ' dataset ')
        pathModels = dsConf.get('pathModels')
        pathPlot = dsConf.get('pathPlot')
        pathDataset = dsConf.get('pathDataset')
        configuration = self.config
        numEsecutions = int(configuration.get('NUM_EXECUTIONS'))
        pathDatasetNumeric = dsConf.get('pathDatasetNumeric')
        pathDatasetEncoded = dsConf.get('pathDatasetEncoded')
        pathDatasetCluster = dsConf.get('pathDatasetCluster')
        testPath = dsConf.get('testPath')
        n_classes = int(configuration.get('N_CLASSES'))

        VALIDATION_SPLIT = float(configuration.get('VALIDATION_SPLIT'))
        N_CLASSES = int(configuration.get('N_CLASSES'))
        pd.set_option('display.expand_frame_repr', False)



        # Preprocessing phase from original to numerical dataset
        PREPROCESSING1 = int(configuration.get('PREPROCESSING1'))

        ds = Datasets(dsConf)
        if (PREPROCESSING1 == 1):
            tic_preprocessing1 = time.time()

            prp.toNumeric(ds)
            toc_preprocessing1 = time.time()
            time_preprocessing1 = toc_preprocessing1 - tic_preprocessing1
            self.file.write("Time Preprocessing: %s" % (time_preprocessing1))

        train_df = pd.read_csv(pathDatasetNumeric + 'Train_standard.csv')
        test_df = pd.read_csv(pathDatasetNumeric + 'Test_standard' + testPath + '.csv')

        self._clsTrain = ds.getClassification(train_df)
        print(self._clsTrain)
        train_X, train_Y = prp.getXY(train_df, self._clsTrain)
        test_X, test_Y = prp.getXY(test_df, self._clsTrain)




        AUTOENCODER_PREP = int(configuration.get('AUTOENCODER_PREPROCESSING'))

        LOAD_AUTOENCODER = int(configuration.get('LOAD_AUTOENCODER'))
        if (AUTOENCODER_PREP == 1):
            tic_preprocessingAutoencoder = time.time()
            if (LOAD_AUTOENCODER == 0):
                autoencoder, best_time, encoder = ah.hypersearch(train_X, train_Y, test_X, test_Y,
                                                                 pathModels + 'autoencoder.h5')

                self.file.write("Time Training Autoencoder: %s" % best_time)


            else:

                print('Load autoencoder')
                autoencoder = load_model(pathModels + 'autoencoder.h5')
                autoencoder.summary()
                encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encod2').output)
                encoder.summary()

            ''' Encoded dataset creation'''
            encoded_train = pd.DataFrame(encoder.predict(train_X))
            encoded_train = encoded_train.add_prefix('feature_')
            encoded_train["classification"] = train_Y
            print(encoded_train.shape)
            print(encoded_train.head())
            encoded_train.to_csv(pathDatasetEncoded + 'train_encoded.csv', index=False)

            encoded_test = pd.DataFrame(encoder.predict(test_X))
            encoded_test = encoded_test.add_prefix('feature_')
            encoded_test["classification"] = test_Y
            print(encoded_test.shape)
            print(encoded_test.head())
            encoded_test.to_csv(pathDatasetEncoded + 'test_encoded' + testPath + '.csv', index=False)
            toc_preprocessingAutoencoder = time.time()
            self.file.write(
                "Time Creations Encoded Dataset: %s" % (toc_preprocessingAutoencoder - tic_preprocessingAutoencoder))




        ORD = int(configuration.get('ORD_DATASET'))
        if (ORD == 1):
            encoded_train, encoded_test = eo.ord(pathDatasetEncoded + 'train_encoded',
                                                 pathDatasetEncoded + 'test_encoded' + testPath)
        else:
            encoded_train = pd.read_csv(pathDatasetEncoded + 'train_encoded_ord.csv')
            encoded_test = pd.read_csv(pathDatasetEncoded + 'test_encoded' + testPath + '_ord.csv')






        image_construction = int(configuration.get('IMAGE'))
        if (image_construction == 1):
            train_df = encoded_train
            test_df = encoded_test
            # divido il train in esempi normali e di attacco, utili per il K-Means
            trainNormal = train_df[train_df['classification'] == 1]
            trainAttack = train_df[train_df['classification'] == 0]
            train_X, train_Y = prp.getXY(train_df, self._clsTrain)
            test_X, test_Y = prp.getXY(test_df, self._clsTrain)
            distTrain, distTest = self.distance(train_X, test_X, trainNormal, trainAttack, self.nearest)
            train_X = np.array(distTrain)
            test_X = np.array(distTest)
            if self.TRAIN == 1:
                saveNpArray(train_X, train_Y, pathDatasetCluster + str(self.num_cluster) + "Train")
            if self.TEST == 1:
                saveNpArray(test_X, test_Y, pathDatasetCluster + str(self.num_cluster) + testPath)

        else:
            print('Load Image dataset')
            train_X = np.load(pathDatasetCluster + str(self.num_cluster) + "TrainX" + '.npy')
            train_Y = np.load(pathDatasetCluster + str(self.num_cluster) + "TrainY" + '.npy')
            test_X = np.load(pathDatasetCluster + str(self.num_cluster) + testPath + "X" + '.npy')
            test_Y = np.load(pathDatasetCluster + str(self.num_cluster) + testPath + "Y" + '.npy')

        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(train_Y, int(n_classes))

        y_test = np_utils.to_categorical(test_Y, int(n_classes))

        print(train_X.shape)
        print(test_X.shape)


        load_cnn = int(configuration.get('LOAD_CNN'))
        if K.image_data_format() == 'channels_first':
            x_train = train_X.reshape(train_X.shape[0], 1, train_X.shape[1], train_X.shape[2])
            x_test = test_X.reshape(test_X.shape[0], 1, train_X.shape[1], train_X.shape[2])
            input_shape = (1, train_X.shape[1], train_X.shape[2])
        else:
            x_train = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
            x_test = test_X.reshape(test_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
            input_shape = (train_X.shape[1], train_X.shape[2], 1)
        XTraining, XValidation, YTraining, YValidation = train_test_split(x_train, y_train, stratify=y_train,
                                                                          test_size=0.2)
        if (load_cnn == 0):

            callbacks_list = [
                callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, restore_best_weights=True),
            ]
            tic = time.time()
            p = ds.getParameters()

            model = Models.CNN(p, input_shape, n_classes)

            history = model.fit(x_train, y_train,
                                batch_size=p['batch_size'],
                                epochs=p['epochs'],
                                callbacks=callbacks_list,
                                verbose=2,
                                #validation_split=VALIDATION_SPLIT,
                                validation_data=(XValidation, YValidation),
                                use_multiprocessing=True,
                                workers=64
                                )

            Plot.printPlotAccuracy(history, 'cnn', dsConf.get('pathPlot'))
            Plot.printPlotLoss(history, 'cnn', dsConf.get('pathPlot'))
            modelName = str(self.num_cluster) + '_' + testPath + 'cnn.h5'
            model.save(pathModels + modelName)

            toc = time.time()
            self.file.write("Time Fitting CNN : " + str(toc - tic))
            self.file.write('\n')

        else:
            print('Load CNN')
            modelName = str(self.num_cluster) + '_' + testPath + 'cnn.h5'
            model = load_model(pathModels + modelName)
            model.summary()


        tic_prediction_classifier = time.time()

        print('Softmax on test set')
        # create pandas for results
        columns = ['Clusters', 'TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
        columnsTemp = ['TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
        results = pd.DataFrame(columns=columns)
        predictions = model.predict(x_test, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        cm = confusion_matrix(test_Y, y_pred)
        r = getResult(cm, n_classes)


        r.insert(0, self.num_cluster)
        dfResults = pd.DataFrame([r], columns=columns)

        print(dfResults)



        results = results.append(dfResults, ignore_index=True)


        toc_prediction_classifier = time.time()
        time_prediction_classifier = (toc_prediction_classifier - tic_prediction_classifier) / numEsecutions
        self.file.write("Time for predictions: %s " % (time_prediction_classifier))
        model.summary()
        results.to_csv(testPath + '_results.csv', index=False)
        self.file.close()



















