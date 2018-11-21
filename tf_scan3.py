import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#N = tf.placeholder(tf.int32, shape=(None,), name='N')
x = tf.placeholder(tf.float32, shape=(None,), name='x')


def fibonacci(last, current):
    return (1 - 0.96) * current + 0.96 * last

square_op = tf.scan(
    fn=fibonacci,
    elems=x,
    initializer=0.0,
)

X = 2 * np.sin(np.linspace(0, 35 * np.pi, 600)) + np.sin(np.linspace(0, 3 * np.pi, 600))
plt.plot(X)
plt.title('Original')
plt.show()

with tf.Session() as session:
    o_val = session.run(square_op, feed_dict={x: X})
    print("Output: ", o_val)

    X2 = np.flip(o_val, axis=0)
    o_val2 = session.run(square_op, feed_dict={x: X2})
    X3 = np.flip(o_val2, axis=0)
    print("Output: ", o_val)


#print(np.flip(o_val, axis=0))
plt.plot(np.sin(np.linspace(0, 3 * np.pi, 600)))
plt.plot(o_val)
plt.plot(o_val2)
plt.show()
