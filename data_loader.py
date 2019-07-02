"""
@Harikrishna_prabhu
June 27-2019
(c) Infinite Analytics
"""
from parameters_keras import DATASET, NETWORK
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

def load_data(validation=False, test=False):
    
    data_dict = dict()
    validation_dict = dict()
    test_dict = dict()
    ohe = OneHotEncoder()

    if DATASET.name == "Fer2013":

        #if train:
        # load train set
        data_dict['X'] = np.load(DATASET.train_folder + '/img_train.npy')
        data_dict['X'] = data_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        if NETWORK.output_size == 7:
            data_dict['Y'] = np.load(DATASET.train_folder + '/label_train.npy')
        elif NETWORK.output_size == 3:
            data_dict['Y'] = np.load(DATASET.train_folder + '/label_train3.npy')
        
        #shuffle
        data_dict['X'], data_dict['Y'] = shuffle(data_dict['X'], data_dict['Y'], random_state=12)
        data_dict['Y'] = data_dict['Y'].reshape(-1, 1)
        data_dict['Y'] = ohe.fit_transform(data_dict['Y'])#
        #data_dict['Y'] = ohe.transform(data_dict['Y'])

        if validation:
            # load validation set
            validation_dict['X'] = np.load(DATASET.validation_folder + '/img_val.npy')
            validation_dict['X'] = validation_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            if NETWORK.output_size == 3:
                validation_dict['Y'] = np.load(DATASET.validation_folder + '/label_val3.npy')
            elif NETWORK.output_size == 7:
                validation_dict['Y'] = np.load(DATASET.validation_folder + '/label_val.npy')
            #shuffle
            validation_dict['X'], validation_dict['Y'] = shuffle(validation_dict['X'], validation_dict['Y'], random_state=32)
            
            validation_dict['Y'] = validation_dict['Y'].reshape(-1, 1)
            validation_dict['Y'] = ohe.transform(validation_dict['Y'])

            if not test:
                return data_dict, validation_dict
             
        
        if test:
            # load test set
            test_dict['X'] = np.load(DATASET.test_folder + '/img_test.npy')
            test_dict['X'] = test_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])

            if NETWORK.output_size == 7:
                test_dict['Y'] = np.load(DATASET.test_folder + '/label_test.npy')
            elif NETWORK.output_size == 3:
                test_dict['Y'] = np.load(DATASET.test_folder + '/label_test3.npy')
            #shuffle
            test_dict['X'], test_dict['Y'] = shuffle(test_dict['X'], test_dict['Y'], random_state=11) 
            test_dict['Y'] = test_dict['Y'].reshape(-1, 1)
            test_dict['Y'] = ohe.transform(test_dict['Y'])

        return data_dict, validation_dict, test_dict

    else:
        print( "Unknown dataset")
        exit()
