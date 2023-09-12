import math


class Scalar:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data  # Data stored in the variable
        self.grad = 0.0  # The gradient of some variable down the computational graph wrt this Scalar
        # A function that will chain (rule) the output's gradient into the input gradients
        self._backward = lambda: None
        # Set of the children nodes from which the Scalar is computed
        self._prev = set(_children)
        self._op = _op  # Operation performed that resulted in the Scalar
        self.label = label  # Name/label of the Scalar

    def __repr__(self):
        return f"Scalar({self.label} = {self.data} | Grad = {self.grad})"

    """
  Logical Operations
  """

    def __gt__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(
            other, label='Constant')  # If constant, wrap it inside a Scalar object
        return True if self.data > other.data else False

    """
  Arithmetic Operations
  """

    def __add__(self, other):

        other = other if isinstance(other, Scalar) else Scalar(
            other, label='Constant')  # If constant, wrap it inside a Scalar object
        out = Scalar(self.data + other.data, (self, other), '+', 'sum')

        def _backward():
            # Take out's grad and propagate it into self's and other's grad
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):

        other = other if isinstance(
            other, Scalar) else Scalar(other, label='Constant')
        out = Scalar(self.data * other.data, (self, other), '*', 'product')

        def _backward():
            self.grad += (other.data) * out.grad
            other.grad += (self.data) * out.grad

        out._backward = _backward

        return out

    # Fallback (reverse mul) if it sees constant * Scalar instead of Scalar * constant
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):

        assert isinstance(other, (int, float)
                          ), "Only supporting int/float powers for now"
        out = Scalar(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad = other * self.data**(other-1) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        x = self.data
        out = Scalar(math.exp(x), (self, ), 'exp', 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def log(self):  # Scalar(log(self.data))
        x = self.data

        out = Scalar(math.log(x), (self, ), 'log', 'log')

        def _backward():
            self.grad = (1/x) * out.grad
        out._backward = _backward

        return out

    """
  Activations
  """

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Scalar(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def topological_sort(self):

        topo = []
        visited = set()

        def build_topo(self):
            if self not in visited:
                visited.add(self)
                for child in self._prev:
                    build_topo(child)
                topo.append(self)

        build_topo(self)

        return topo

    def backward(self):

        self.grad = 1
        topo = self.topological_sort()

        for node in reversed(topo):
            node._backward()
