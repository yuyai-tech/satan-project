
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


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

test_data = data[1][0]
test_labels = data[1][1]

print(train_data.shape)
print(train_labels.shape)

number_of_cats = np.max(train_labels) + 1
print(number_of_cats)

# multi cat labels
train_labels_len = train_labels.shape[0]
train_labels_reshaped = train_labels.reshape([train_labels_len, 1])

mlb = MultiLabelBinarizer()
multi_cat_binary_labels = mlb.fit_transform(train_labels_reshaped)




item_data = train_data[530]
item_label = train_labels[530]
print(item_data.shape)

plt.imshow(item_data)
plt.title('LABEL: ' + str(item_label))
plt.show()


# train data
X = train_data
Y = (train_labels == 5).astype(int)

print(X.shape)
print(Y.shape)

# model

model = Sequential()
model.add(
    Input(shape=(28, 28, ))
)
model.add(
    Flatten()
)
model.add(
    Dense(5, activation='relu')
)
model.add(
    Dense(1, activation='sigmoid')
)
model.summary()


# train model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=3, batch_size=32, verbose=1)







