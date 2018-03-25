import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    #layer_name = 'layer{}'.format(n_layer)
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/Weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name+'/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs

#define the placeholder for in puts to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

#add layer
l1 = add_layer(xs, 1, 10, layer_name='hiden_layer', activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, layer_name='output_layer', activation_function=None)

#error between prediciotn and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#train data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph) #写入所有的tf.name_scope

    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, i) # 向writer中加入迭代的一些结果
                          
# # $ tensorboard --logdir logs
    
