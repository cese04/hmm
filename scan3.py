import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

X = 2 * np.random.randn(300) + np.sin(np.linspace(0, 3 * np.pi, 300))
plt.plot(X)
plt.title('Original')
plt.show()

decay = T.scalar('decay')
sequence = T.vector('sequence')


def recurrence(x, last, decay):
    return (1-decay)*x + decay*last


outputs, updates = theano.scan(
    fn=recurrence,
    sequences=sequence,
    n_steps=sequence.shape[0],
    outputs_info=np.array([0], np.float64),
    non_sequences=[decay],
)

lpf = theano.function(
    inputs=[sequence, decay],
    outputs=outputs,
    allow_input_downcast=True
)

Y = lpf(np.float64(X), np.float64(0.99))
plt.plot(Y)
plt.title('filtered')
plt.show()
