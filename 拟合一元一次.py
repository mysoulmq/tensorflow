import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.2 + 0.4


###Create tensorflow structure start###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)
###Create tensorflow structure end###
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train)
        if i % 50 == 0:
            print(i, sess.run(Weights), sess.run(biases))
            
