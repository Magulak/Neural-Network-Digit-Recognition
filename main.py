import tensorflow as tf
import matplotlib.pyplot as plt

# https://www.youtube.com/watch?v=wQ8BIBpya2k

mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits 0-9
(x_train, y_train), (x_test, y__test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)  # convert values from 0-255 to 0-1
y_train = tf.keras.utils.normalize(y_train, axis=1)  # it will make data easier to read for neural network(idk why)
model = tf.keras.models.Sequental()
model.add()
# print(x_train[0])
plt.imshow(x_train[1], cmap=plt.cm.binary)
plt.show()
