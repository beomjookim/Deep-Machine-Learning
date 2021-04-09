'''
Downloading TF and doing some tutorials on Python IDLE

I already have Python downloaded on my Desktop so download TensorFlow first.
do 'pip3 install tensorflow' on CMD and it will do.

when I compiled a simple code it showed an error with "Could not find the DLL(s) 'msvcp140.dll or msvcp140_1.dll'" line.
so I downloaded required files from Visual Studio.

then different error showed, "module 'tensorflow' has no attribute 'Session'"
this was because the Session syntax has been changed on the new TF version.
tf.Session() is used in TF ver. 1.x.x. but mine is 3.9.2.(used tf.__version__)
in 3.9.2, there's no Session() required.
'''


# code:

import numpy as np
import tensorflow as tf

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# we use model called Sequential() which is in keras library.
# Sequential() is good for one-input, one-output general layer stack.
tf.model = tf.keras.Sequential()

# units == output shape, input_dim == input shape
# keras.layers.Dense is a non-viewable process in between input and output.
# units is the dim of the output space, input_dim is the dim of the input space.
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(learning_rate = 0.1)  # SGD == Standard Gradient Descendent
tf.model.compile(loss='mse', optimizer=sgd)         # mse == mean_squared_error, 1/m * sig (y'-y)^2

# prints summary of the model
tf.model.summary()

# fit() executes training
# one epoch is when an entire dataset is passed forward and backward thru the neural network
tf.model.fit(x_train, y_train, epochs=200)

# predict() returns predicted value
y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)
