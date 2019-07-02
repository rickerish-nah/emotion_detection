"""
@Harikrishna_prabhu
June 27-2019
(c) Infinite Analytics
"""
import os
###
class Dataset:
    name = 'Fer2013'
    train_folder = 'data/Fer2013/Train'
    validation_folder = 'data/Fer2013/Valid'
    test_folder = 'data/Fer2013/Test'
    #shape_predictor_path='shape_predictor_68_face_landmarks.dat'
###
class Network:
    model = 'H'
    input_size = 48
    output_size = 3
    activation = 'relu'
    loss = 'categorical_crossentropy'
    use_batchnorm_after_conv_layers = True
    use_batchnorm_after_fully_connected_layers = False
###
class Hyperparams:
    dropout = 0.25#keep_prob = 0.75   # dropout = 1 - keep_prob
    learning_rate = 0.0001
    learning_rate_decay = 1e-6
    optimizer = 'adam'  # {'momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta'}
    optimizer_param = 0.9   # (i) beta1 value for Adam or (ii) momentum value for Momentum optimizer 


class Training:
    batch_size = 128
    epochs = 50
    #snapshot_step = 500
    #vizualize = True
    logs_dir = "logs"
    checkpoint_dir = "checkpoints/chk"
    best_checkpoint_path = "checkpoints/best/"
    max_checkpoints = 5
    checkpoint_frequency = 0.2 # in hours
    save_model = True
    save_model_path = "Model/"

class Predictor:
    emotions = ["Negative", "Positive", "Neutral"]
    print_emotions = True
    camera_source = 0
    face_detection_classifier = "lbpcascade_frontalface.xml"
    show_confidence = False
    time_to_wait_between_predictions = 0.5

class OptimizerSearchSpace:
    learning_rate = {'min': 0.00001, 'max': 0.1}
    learning_rate_decay = {'min': 0.5, 'max': 0.99}
    optimizer = ['momentum']   # ['momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta']
    optimizer_param = {'min': 0.5, 'max': 0.99}
    keep_prob = {'min': 0.7, 'max': 0.99}

def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

DATASET = Dataset()
NETWORK = Network()
TRAINING = Training()
HYPERPARAMS = Hyperparams()
PREDICTOR = Predictor()
OPTIMIZER = OptimizerSearchSpace()

make_dir(TRAINING.logs_dir)
make_dir(TRAINING.checkpoint_dir)

