import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2, 3],
                       [3, 3]])
output = matrix1 * matrix2 #矩阵点乘
##output = tf.matmul(matrix1, matrix2) #矩阵乘法

#method1
##sess = tf.Session()
##result = sess.run(output)
##print(result)
##sess.close()

#method2
with tf.Session() as sess:
    print(sess.run(output))
