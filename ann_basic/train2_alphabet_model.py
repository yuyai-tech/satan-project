import os
os.chdir("./ann_basic/")

import matplotlib as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense
data = from ann_basic.data.data import

# load data


# pre pros data

item_data = data[]

dog_list = ["tasha", "coopy", "firualais", "quilla", "melo", "tarzan", "boby"]
cat_list = ["maya", "perlita", "tutu", "spyke", "teodoro", "pancito", "chanchirri"]

for dog in dog_list:
    dog_with_last_name = dog + "diana"
    dog_list_last_name.append(dog_with_last_name)