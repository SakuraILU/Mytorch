import graphviz
import math


class Value():
    def __init__(self, data, childs=set(), op="alloc", label=""):
        self.__data = data
        self.label = label

        self.__childs = childs
        self.__op = op

        self.grad = 0
        self.__backward = lambda: None

    def item(self):
        return self.__data

    def childs(self):
        return self.__childs

    def backward(self):
        self.grad = 1.0

        visited = []
        ordered = []

        def topsort(node):
            visited.append(node)
            for v in node.childs():
                if v in visited:
                    continue
                topsort(v)
            ordered.append(node)

        topsort(self)

        for v in reversed(ordered):
            v.__backward()

    def visialize(self):
        edges = []
        nodes = []
        visited = []

        def build_graph(node):
            visited.append(node)
            nodes.append(node)
            for v in node.childs():
                edges.append((node, v))
                if v in visited:
                    continue
                build_graph(v)
        build_graph(self)

        h = graphviz.Digraph('value_graph', format='png',
                             graph_attr={'rankdir': "TB"})
        for v in nodes:
            label = ""
            if v.__op is not None:
                label += f"{v.__op} |"
            label += "{" + f"data {v.__data:.4f} | grad {v.grad:.4f}" + "}"
            if v.label is not None:
                label += "|" + v.label
            h.node(name=str(id(v)), label=label, shape="record")

        for e in edges:
            v0 = e[0]
            v1 = e[1]
            h.edge(str(id(v0)), str(id(v1)))
        h.render('value_graph')

    def log(self):
        data = math.log(self.__data)
        out = Value(data, (self, ))
        out.__op = "log"

        def backward():
            self.grad += out.grad * 1 / self.__data

        out.__backward = backward

        return out

    def exp(self):
        data = math.exp(self.__data)
        out = Value(data, (self, ))
        out.__op = "exp"

        def backward():
            self.grad += out.grad * data

        out.__backward = backward

        return out

    def tanh(self):
        data = math.tanh(self.__data)
        out = Value(data, (self, ))
        out.__op = "tanh"

        def backward():
            self.grad += out.grad * (1 - data ** 2)

        out.__backward = backward

        return out

    def relu(self):
        data = self.__data if self.__data >= 0 else 0
        out = Value(data, (self, ))
        out.__op = "relu"

        def backward():
            self.grad += out.grad * self.__data if self.__data > 0 else 0

        out.__backward = backward

        return out

    def __add__(self, other):
        other = other if isinstance(
            other, Value) else Value(other, label="num")

        data = self.__data + other.__data
        out = Value(data, (self, other))
        out.__op = "+"

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.__backward = backward

        return out

    def __sub__(self, other):
        other = other if isinstance(
            other, Value) else Value(other, label="num")

        data = self.__data - other.__data
        out = Value(data, (self, other))
        out.__op = "-"

        def backward():
            self.grad += out.grad
            other.grad += -out.grad

        out.__backward = backward

        return out

    def __rsub__(self, other):
        other = other if isinstance(
            other, Value) else Value(other, label="num")
        return other - self

    def __radd__(self, other):
        other = other if isinstance(
            other, Value) else Value(other, label="num")
        return other + self

    def __mul__(self, other):
        other = other if isinstance(
            other, Value) else Value(other, label="num")

        data = self.__data * other.__data
        out = Value(data, (self, other))
        out.__op = "*"

        def backward():
            self.grad += out.grad * other.__data
            other.grad += out.grad * self.__data

        out.__backward = backward

        return out

    def __rmul__(self, other):
        other = other if isinstance(
            other, Value) else Value(other, label="num")
        return other * self

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        assert isinstance(other, (float, int)
                          ), "only support value ** float/int"

        # in this code, set other to be Value...just for visualization and debugging...
        # actually, other is always treated like a const float, so no grad for it
        other = Value(other, label="num")
        data = self.__data ** other.__data
        out = Value(data, (self, other))
        out.__op = "**"

        def backward():
            self.grad += out.grad * other.__data * \
                self.__data ** (other.__data - 1)
            other.grad = 0

        out.__backward = backward

        return out

        # return (other * self.log()).exp() # esay but ineffient...it's a combination of multiple operations

    def __truediv__(self, other):
        other = other if isinstance(
            other, Value) else Value(other, label="num")

        data = self.__data / other.__data
        out = Value(data, (self, other))
        out.__op = "/"

        def backward():
            self.grad += out.grad * 1 / other.__data
            other.grad += out.grad * self.__data * (-1) * other.__data ** (-2)

        out.__backward = backward

        return out

        # easy to handle, but it is a combination of two operations...not effience
        # return self * other ** (-1)

    def __rtruediv__(self, other):
        other = other if isinstance(
            other, Value) else Value(other, label="num")

        return other / self

    def __iadd__(self, other):
        other = other.__data if isinstance(other, Value) else other
        self.__data += other
        return self

    def __isub__(self, other):
        other = other.__data if isinstance(other, Value) else other
        self.__data -= other
        return self

    def __str__(self):
        return f"Value(data = {self.__data:.4f}\t op = {self.__op}\t lable = {self.label} \t grad = {self.grad:.4f})"
