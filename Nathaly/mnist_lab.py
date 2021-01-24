import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

data = tf.keras.datasets.mnist.load_data()

train_data = data[0][0]
train_labels = data[0][1]

print(train_data.shape)
print(train_labels.shape)


item_data = train_data[300]
item_label = train_labels[300]
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
    Dense(3, input_dim=input_dim, activation='relu')
)
model.add(
    Dense(3, activation='relu')
)
model.add(
    Dense(1, activation='sigmoid')
)

model.summary()


# train model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=20, batch_size=256, verbose=1)



## assess train data

prediction_train = model.predict(X)

auc = roc_auc_score(Y, prediction_train)

print('Train AUC: %.f' % auc)

fpr, tpr, thresholds = roc_curve(Y, prediction_train)
plot_roc_curve(fpr, tpr)

## Data test
test_data = data[1][0]
test_labels =