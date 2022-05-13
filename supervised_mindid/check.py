import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers
from sklearn.manifold import TSNE

from backbones import *
from data_load import *

frame_size   = 128
num_classes  = 109
sessions = [1,2,3,4,5,6]
path = os.path.join(os.getcwd(), '..', 'mindid_dataset', "files")

x_train, y_train = process_data(path, sessions=sessions, num_class=num_classes, frame_size = frame_size)