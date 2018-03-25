import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    W_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function == None:
        outputs = W_plus_b
    else:
        outputs = activation_function(W_plus_b)
    return outputs

#Make some real data
x = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x.shape)
y = np.square(x) + 0.5 + noise

#define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1]) #None就是说行数不一定
ys = tf.placeholder(tf.float32, [None, 1])
#add hidden layer
in_size = 1
hid_size = 10
out_size = 1
l1 = add_layer(xs, in_size, hid_size, tf.nn.relu)
#add output layer
prediction = add_layer(l1, hid_size, out_size)

# 预测值和真实值的误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys), 1), 0)#每行相加然后每列求和
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

#plot the real data
fig = plt.figure()
ax = fig.add_subplot(111) #前两个参数将图分为x*y。第三个代表第几个小图
ax.scatter(x, y) #把各点描出来而不连接
plt.ion() # 交互绘图功能
plt.show() #  显示所绘制的图形

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={xs:x, ys:y})
        if i % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs:x, ys:y})
            #plot the prediction
            lines = ax.plot(x, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
