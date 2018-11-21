import numpy as np
import tensorflow as tf

N = tf.placeholder(tf.int32, shape=(None,), name='N')


def fibonacci(last, current):
    return [last[1], last[0]+last[1]]

square_op = tf.scan(
	fn=fibonacci,
	elems=N,
	initializer=[0, 1],
)

with tf.Session() as session:
	o_val = session.run(square_op, feed_dict={N:np.arange(20)})
	print("Output: ", o_val)