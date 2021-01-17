
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = tf.keras.datasets.mnist.load_data()

train_data = data[0][0]
train_labels = data[0][1]

print(train_data.shape)
print(train_labels.shape)


item_data = train_data[530]
item_label = train_labels[530]
print(item_data.shape)

plt.imshow(item_data)
plt.title('LABEL: ' + str(item_label))
plt.show()


# creating labels
binary_labels = []
for value in train_labels:
    # print(value)
    if value == 0:
        binary_labels.append(1)
    else:
        binary_labels.append(0)

binary_labels_array = np.array(binary_labels)

# flatten data
train_data_flatten = []
for value in train_data:
    value_flatten = value.flatten()
    train_data_flatten.append(value_flatten)

train_data_flatten_array = np.array(train_data_flatten)

# train data
X = train_data_flatten_array
Y = binary_labels_array

print(X.shape)
print(Y.shape)
