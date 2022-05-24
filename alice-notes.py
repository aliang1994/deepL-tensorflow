import keras
import numpy as np
import tensorflow as tf

"""
Course 1
"""

"""
W1: fitting linear functions
"""
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0, 20.0]))
print(tf.__version__)


"""
W2: fashion MNIST
"""

f_mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = f_mnist.load_data()

# Build the classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# 28 x 28 pixels, 0-255
# for image / pixels, normalize first
# Flatten: turning 28 x 28 into 784 x 1 array
# improvements: 1. adding another Dense layer; 2. adding more neurons in Dense layer; 3. increase epoch 
# be aware of overfitting -  loss value stops decreasing, and sometimes increases

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50)