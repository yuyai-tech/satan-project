import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from ann_basic.data.data import *

## Data

data_image = [ a_image, a1_image, a2_image, a3_image, b_image, c_image, d_image,
               e_image, e1_image, e2_image, f_image, g_image, h_image, i_image,
               j_image, k_image, l_image, m_image, n_image, o_image, p_image,
               q_image, r_image, s_image, t_image, u_image, v_image, w_image,
               x_image, y_image, z_image ]

## data array

data_image_array = []
for image in data_image:
    image_array = np.array(image)
    data_image_array.append(image_array)

plt.imshow(a_image)
plt.show()

## data flatten

data_image_flatten = []
for image in data_image_array:
    image_flatten = image.flatten()
    data_image_flatten.append(image_flatten)

data_image_flatten_array = np.array(data_image_flatten)

print(data_image_flatten_array.shape)

## data labels

labels = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

print(labels.shape)

## create model

input_dim = 10*11
X = data_image_flatten_array
Y = labels

model = Sequential()
model.add(Dense(50, input_dim=input_dim, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# train model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=20, batch_size=1, verbose=1)

# test
model.predict(data_image_flatten_array)

prediction_first = model.predict(data_image_flatten_array)


