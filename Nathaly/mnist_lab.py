import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# model
input_dim = X.shape[1]
model = Sequential()
model.add(
    Dense(50, input_dim=input_dim, activation='relu')
)
model.add(
    Dense(50, activation='relu')
)
model.add(
    Dense(50, activation='relu')
)
model.add(
    Dense(1, activation='sigmoid')
)

model.summary()


# train model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=20, batch_size=256, verbose=1)

