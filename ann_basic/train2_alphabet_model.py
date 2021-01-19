import os
os.chdir("./ann_basic/")

import matplotlib as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense
from ann_basic.data.data import