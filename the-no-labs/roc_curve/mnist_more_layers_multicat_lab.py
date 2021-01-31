
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix
import seaborn as sns


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


# train data
X = train_data
Y = multi_cat_binary_labels

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
    Dense(300, activation='relu')
)
model.add(
    Dense(300, activation='relu')
)
model.add(
    Dense(10, activation='sigmoid')
)
model.summary()


# train model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=10, batch_size=256, verbose=1)



# test labels
test_labels_len = test_labels.shape[0]
test_labels_reshaped = test_labels.reshape([test_labels_len, 1])

mlb = MultiLabelBinarizer()
multi_cat_binary_labels_test = mlb.fit_transform(test_labels_reshaped)


X_test = test_data
Y_test = multi_cat_binary_labels_test

prediction_test = model.predict(X_test)
prediction_test_argmax = np.argmax(prediction_test, axis=1)

cm = confusion_matrix(Y_test, prediction_test_argmax)
sns.heatmap(cm, annot=True, fmt="d", cmap="Spectral")
plt.show()


