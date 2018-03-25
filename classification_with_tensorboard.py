import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram('Weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram('biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function == None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram('outputs', outputs)
    return outputs

def compute_accuracy(v_xs, v_ys):
     global prediction
     y_pre = sess.run(prediction, feed_dict={xs: v_xs})
     correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
     return result
    

#define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 28*28], name='x_data')
    ys = tf.placeholder(tf.float32, [None, 10], name='y_data')

#add layer
in_size = 28 * 28
##hid_size = 88
out_size = 10
##l1 = add_layer(xs, in_size, hid_size, activation_function=tf.sigmoid)
prediction = add_layer(xs, in_size, out_size, activation_function=tf.nn.softmax)

#the error between prediction and real data
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                           reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs_class', sess.graph)
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        '''
        该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，
        然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
        '''
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            accuracy = compute_accuracy(mnist.test.images, mnist.test.labels)
            result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys})
            writer.add_summary(result, i)
            print(accuracy)

