import numpy as np

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

## data flatten

data_image_flatten = []
for image in data_image_array:
    image_flatten = image.flatten()
    data_image_flatten.append(image_flatten)

data_image_flatten_array = np.array(data_image_flatten)

print(data_image_flatten_array.shape)

## data labels

