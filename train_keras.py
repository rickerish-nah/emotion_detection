"""
@Harikrishna_prabhu
June 27-2019
(c) Infinite Analytics
"""
import time
import argparse
import os
import numpy as np
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint

from data_loader import load_data 
from parameters_keras import DATASET, TRAINING, HYPERPARAMS, NETWORK
from model_keras import build_model


def train(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param, 
        learning_rate=HYPERPARAMS.learning_rate, dropout=HYPERPARAMS.dropout, 
        learning_rate_decay=HYPERPARAMS.learning_rate_decay,epochs=TRAINING.epochs,
        batch_size=TRAINING.batch_size,train_model=True):

        print( "loading dataset " + DATASET.name + "...")

        data, validation  = load_data(validation=True)
        assert(data['X'].shape[0] == data['Y'].shape[0])
        assert(validation['X'].shape[0] == validation['Y'].shape[0])
        #print(validation['Y'])
        #build model
        model = build_model()
        # Training phase
        print( "start training...")
        print( "  - emotions = {}".format(NETWORK.output_size))
        print( "  - model = {}".format(NETWORK.model))
        print(model.summary())
        print( "  - optimizer = '{}'".format(optimizer))
        print( "  - learning_rate = {}".format(learning_rate))
        print( "  - learning_rate_decay = {}".format(learning_rate_decay))
        print( "  - otimizer_param ({}) = {}".format('beta1' if optimizer == 'adam' else 'momentum', optimizer_param))
        print( "  - dropout = {}".format(dropout))
        print( "  - epochs = {}".format(TRAINING.epochs))
        #print( "  - use batchnorm after conv = {}".format(NETWORK.use_batchnorm_after_conv_layers))
        #print( "  - use batchnorm after fc = {}".format(NETWORK.use_batchnorm_after_fully_connected_layers))
        print( "Train Size:", data['X'].shape,data['Y'].shape)
        print( "Val Size:", validation['X'].shape,validation['Y'].shape)

        start_time = time.time()
        # checkpoint
        filepath=TRAINING.save_model_path+"weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        model_info = model.fit(data['X'], data['Y'],
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation['X'], validation['Y']),callbacks=[checkpoint])

        training_time = time.time() - start_time
        print( "training time = {0:.1f} sec".format(training_time))

        if TRAINING.save_model:
                print( "saving model...")
                model.save(TRAINING.save_model_path+"model.h5")
                # if not(os.path.isfile(TRAINING.save_model_path)) and os.path.isfile(TRAINING.save_model_path + ".meta"):
                #         os.rename(TRAINING.save_model_path + ".meta", TRAINING.save_model_path)

        print( "evaluating...")
        valid_y = model.predict(validation['X'])
        print(classification_report(np.argmax(validation['Y'],axis=1),np.argmax(valid_y,axis=1)))
        validation_accuracy = model.evaluate(validation['X'], validation['Y'])
        print( "  - validation accuracy = ", (sum(validation_accuracy)/len(validation_accuracy)*100))

def evaluate():
    print( "loading dataset " + DATASET.name + "...")

    __, validation, test = load_data(validation=True, test=True)

    assert(validation['X'].shape[0] == validation['Y'].shape[0])
    assert(test['X'].shape[0] == test['Y'].shape[0])
    # Testing phase : load saved model and evaluate on test dataset

    model = build_model()
    #model.summary()
    print( "start evaluation...")
    print( "loading pretrained model...")
    try:
        model.load_weights(TRAINING.save_model_path+'model.h5')
    except:
        print('Failed to load weights from - ',TRAINING.save_model_path+'model.h5')
        exit()
    print( "--")
    print( "Validation samples: {}".format(validation['Y'].shape[0]))
    print( "Test samples: {}".format(test['Y'].shape[0]))
    print( "--")
    print( "evaluating...")
    start_time = time.time()
    valid_y = model.predict(validation['X'])
    print(classification_report(np.argmax(validation['Y'],axis=1),np.argmax(valid_y,axis=1)))
    #validation_accuracy = model.evaluate(validation['X'], validation['Y'])
    #print( "  - validation accuracy = ", (sum(validation_accuracy)/len(validation_accuracy)*100))
    test_y = model.predict(test['X'])
    print(classification_report(np.argmax(test['Y'],axis=1),np.argmax(test_y,axis=1)))
    #test_accuracy = model.evaluate(test['X'], test['Y'])
    #print( "  - test accuracy = ", (sum(test_accuracy)/len(test_accuracy)*100))
    print( "  - evalution time = {0:.1f} sec".format(time.time() - start_time))

# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default="no", help="if 'yes', launch training from command line")
parser.add_argument("-e", "--evaluate", default="no", help="if 'yes', launch evaluation on test dataset")
parser.add_argument("-m", "--max_evals", help="Maximum number of evaluations during hyperparameters search")
args = parser.parse_args()
if args.train=="yes" or args.train=="Yes" or args.train=="YES":
        train()
if args.evaluate=="yes" or args.evaluate=="Yes" or args.evaluate=="YES":
        evaluate()

