import random
import numpy as np

from network import MLP
from engine import Value

if __name__ == "__main__":
    # some test code...

    # train...
    # train dataset
    xs_pos = 2 * (np.random.rand(20, 5) + 0.5)
    ys_pos = np.ones(20)  # label 1
    xs_neg = 2 * (np.random.rand(20, 5) - 0.5)
    ys_neg = -np.ones(20)  # label -1
    xs = np.vstack((xs_pos, xs_neg))
    ys = np.hstack((ys_pos, ys_neg))

    lr = 0.01
    mlp = MLP(5, [10, 30, 1], activate="tanh")
    for _ in range(15):
        # calculate all y for xs-->network
        y = []
        for x_s in xs:
            y += mlp(x_s)  # note: mlp output a list...
        loss = sum((y_s - y) ** 2 for y, y_s in zip(y, ys))/len(ys)

        loss.backward()
        for param in mlp.parameters():
            param -= lr * param.grad
        mlp.zero_grad()

        print(f"[Loss]: {loss.item() : .4f}")

    # test...
    # test dataset
    xt_pos = 2 * (np.random.rand(20, 5) + 0.5)
    yt_pos = np.ones(20)
    xt_neg = 2 * (np.random.rand(20, 5) - 0.5)
    yt_neg = -np.ones(20)
    xt = np.vstack((xt_pos, xt_neg))
    yt = np.hstack((yt_pos, yt_neg))

    loss = Value(0)
    for x_t, y_t in zip(xt, yt):
        y = mlp(x_t)
        print(x_t, y[0].item())
        loss += (y[0] - y_t) ** 2
    loss = loss / len(yt)
    print(f"[TEST LOSS]: {loss.item() :.4f}")
