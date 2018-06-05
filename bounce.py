import numpy as np

transitions = {}
row_sums = {}

# collect counts
for line in open('site_data.csv'):
    s, e = line.rstrip().split(',')
    transitions[(s, e)] = transitions.get((s, e), 0.0) + 1
    row_sums[s] = row_sums.get(s, 0.0) + 1


# Normalize
for k, v in transitions.items():
    s, e = k
    transitions[k] = v / row_sums[s]

print("Initial state distribution")
for k, v in transitions.items():
    s, e = k
    if s == '-1':
        print(e, v)

# Which has the highest bounce
for k, v in transitions.items():
    s, e = k
    if e == 'B':
        print('Bounce rate for %s: %s' % (s, v))