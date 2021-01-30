
import matplotlib.pylab as plt
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub.keras_layer import KerasLayer
import numpy as np
import PIL.Image as Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# load images
data_set_directory = "./the-no-labs/easy_image_recognition_lab/data"

IMAGE_SHAPE = (128, 128)
image_generator = ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(data_set_directory, target_size=IMAGE_SHAPE)


# print image
n_image_to_show = 5
data_to_show = image_data[0][0][n_image_to_show]
label_to_show = image_data[0][1][n_image_to_show]
plt.imshow(data_to_show)
plt.title('LABEL: ' + str(label_to_show) + ' indices ' + str(image_data.class_indices))
plt.show()



