import random
from abc import ABC, abstractmethod


from engine import Value


class Module(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def parameters(self, x):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    def forward(self, x):
        return self(x)


class Neuron(Module):
    def __init__(self, nin):
        self.__nin = nin
        self.__w = [Value(random.uniform(-1, 1),
                          label=f"w{i}") for i in range(nin)]
        self.__b = Value(random.uniform(-1, 1), label="b")

    def __call__(self, x):
        out = sum([w * x for x, w in zip(x, self.__w)], self.__b)
        return out

    def parameters(self):
        return self.__w + [self.__b]

    def zero_grad(self):
        for w in self.__w:
            w.grad = 0
        self.__b.grad = 0


class Linear(Module):
    def __init__(self, nin, nout):
        self.__neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.__neurons]

    def parameters(self):
        return [param for neuron in self.__neurons for param in neuron.parameters()]

    def zero_grad(self):
        for neuro in self.__neurons:
            neuro.zero_grad()


class Tanh(Module):
    def __init__(self):
        pass

    def __call__(self, x):
        return [v.tanh() for v in x]

    def parameters(self):
        return []

    def zero_grad(self):
        return


class Relu(Module):
    def __init__(self):
        pass

    def __call__(self, x):
        return [v.relu() for v in x]

    def parameters(self):
        return []

    def zero_grad(self):
        return


class Layer(Module):
    def __init__(self, nin, nout, activate="tanh"):
        assert activate in ("", "tanh", "relu"), "only support tanh/relu"
        self.__liner = Linear(nin, nout)
        self.__activate = Tanh() if activate == "tanh" else Relu()

    def __call__(self, x):
        x = self.__liner(x)
        return self.__activate(x)

    def parameters(self):
        return self.__liner.parameters()

    def zero_grad(self):
        self.__liner.zero_grad()


class MLP(Module):
    def __init__(self, nin, nouts, activate="tanh"):
        assert isinstance(
            nouts, list), "nouts should be a list, even just one layer"
        ns = [nin] + nouts
        self.__layers = [Layer(ns[i], ns[i+1], activate=activate)
                         for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.__layers:
            x = layer(x)
        return x

    def parameters(self):
        return [param for layer in self.__layers for parm in layer.parameters()]

    def zero_grad(self):
        for layer in self.__layers:
            layer.zero_grad()
