import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def add_layer(inputs, in_size, out_size, layer_name=None,
              activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 8*8])
ys = tf.placeholder(tf.float32, [None, 10])

# add layer
in_size = 8 * 8
hid_size = 50
out_size = 10
l1 = add_layer(xs, in_size, hid_size, 'hid_layer', tf.nn.tanh)
prediction = add_layer(l1, hid_size, out_size, 'out_layer', tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                       reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(0.5)
train_step = optimizer.minimize(cross_entropy)

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('./logs/test', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
        if i % 50 == 0:
            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)


    
