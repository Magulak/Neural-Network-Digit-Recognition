import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# https://www.youtube.com/watch?v=wQ8BIBpya2k

mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)  # convert values from 0-255 to 0-1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # it will make data easier to read for neural network(idk why)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())  # 28x28[2D] needs to be converted to [1D]
model.add(
    tf.keras.layers.Dense(2560, activation=tf.nn.relu))  # Use nn.relu as default and then change it to other layers
model.add(tf.keras.layers.Dense(2560, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # Use softmax for probability

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)  # epochs =  how many times train over with that dataset

val_loss, val_acc = model.evaluate(x_test, y_test)
model.save('num_reader')
print(val_loss, val_acc)
predictions = model.predict([x_test])

print(np.argmax(predictions[0]))
# print(x_train[0])
# plt.imshow(x_train[1], cmap=plt.cm.binary)
# plt.show()
