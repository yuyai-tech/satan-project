
import tensorflow as tf
import matplotlib.pyplot as plt


# data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# plot
data_to_plot = x_train[2]
label_to_plot = y_train[2]
plt.imshow(data_to_plot)
plt.title('LABEL: ' + str(label_to_plot))
plt.colorbar()
plt.show()

