import tensorflow as tf

state = tf.Variable(0, name='counter') #tensorflow.python.ops.variables.Variable
#print(state.name)
one = tf.constant(1)
#one = 1 #直接用这个也可

new_value = tf.add(state, one) #new_value是tensorflow.python.framework.ops.Tensor
                               #即，tf.constant所对应的类型
##print(type(new_value), type(state), type(one))
update= tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
