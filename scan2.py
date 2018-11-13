import numpy as np
import theano
import theano.tensor as T

N = T.iscalar('N')


def rec(n, fn_1, fn_2):
    return fn_1 + fn_2, fn_1


outputs, updates = theano.scan(
    fn=rec,
    sequences=T.arange(N),
    n_steps=N,
    outputs_info=[1, 1]
)

fibonacci = theano.function(
    inputs=[N],
    outputs=outputs
)

o_val = fibonacci(10)

print("Output: ", o_val)