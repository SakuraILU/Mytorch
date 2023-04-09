import random
import numpy as np

from network import MLP

if __name__ == "__main__":
    # some test code...

    # layer = Layer(5, 3, activate="tanh")
    mlp = MLP(5, [3, 3])
    xs = [1, 2, 3, 4, 5]
    ys = [0.5, 1, 1.5, 2, 2.5]

    y = mlp(xs)

    loss = sum((ys - y)**2 for ys, y in zip(ys, y))/5
    loss.backward()
    mlp.zero_grad()

    loss.visialize()
