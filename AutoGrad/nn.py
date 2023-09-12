from typing import List
import random
from AutoGrad.engine import Scalar


class Neuron:

    def __init__(self, nin: int, label='',
                 initializer=random.uniform,
                 activation: str = 'tanh'
                 ):
        self.n_inputs = nin
        self.w = [Scalar(initializer(-1, 1)) for _ in range(nin)]
        self.b = Scalar(initializer(-1, 1))
        self.activation = activation
        self.label = label

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.label} => Inputs: {self.n_inputs} | Activation: {self.activation}"

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = getattr(act, self.activation)()
        out.label = self.label+' output'
        return out


class Layer:

    def __init__(self, nin, nout, label='',
                 initializer=random.uniform,
                 activation: str = 'tanh'
                 ):
        self.label = label
        self.n_inputs = nin
        self.neurons = [Neuron(
            nin, label=f'{self.label} Neuron_{i}', activation=activation) for i in range(nout)]

    def __repr__(self):
        return f"{self.label} => Inputs: {self.n_inputs} | Units: {len(self.neurons)}"

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs


class MLP:

    def __init__(self, nin, nouts: List,
                 initializers=None,
                 activations=None
                 ):
        activations = ['tanh' for _ in range(
            len(nouts))] if activations == None else activations
        initializers = [random.uniform for _ in range(
            len(nouts))] if initializers == None else initializers
        n = [nin]+nouts
        self.layers = [Layer(
            n[i], n[i+1], f'Layer_{i+1}', initializers[i], activations[i]) for i in range(len(n)-1)]

    def __repr__(self):
        return '\n'.join(map(str, self.layers))

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
