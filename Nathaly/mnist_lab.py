import tensorflow as tf
import matplotlib.pyplot as plt

data = tf.keras.datasets.mnist.load_data()

train_data = data [0][0]
train_labels = data[0][1]
print(train_data.shape)
print(train_labels.shape)

item_data = train_data[100]
item_labels = train_labels[100]
print(item_data)

plt.imshow(item_data)
plt.title('LABEL:' + str(item_labels))
plt.show()

data [0]


