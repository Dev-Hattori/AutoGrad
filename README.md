
# AutoGrad
---
This is an implementation of **PyTorch**esque **Autograd** engine.
Instead of *tensors* it works on *scalars* and implements **backpropagation** over *Directed Acyclic Graphs* operating over these *scalars*.

## Stuff in it
### `Scalar`
This is the main data-type that is used to to implement the computational graphs.
#### Example Usage

##### Initialization and Arithmetic Operations
```python
a = Scalar(2.0, label='a')
b = Scalar(3.0, label='b')
a + b, a - b, a + 2, 2 - b 
```

`(Scalar(a+b = 5.0 | Grad = 0.0),
 Scalar(a+b*Const(-1) = -1.0 | Grad = 0.0),
 Scalar(a+Const(2) = 4.0 | Grad = 0.0),
 Scalar(b*Const(-1)+Const(2) = -1.0 | Grad = 0.0))`

```python
a * 2, 2 * a, a / 2, 2 / a
```

`(Scalar(a*Const(2) = 4.0 | Grad = 0.0),
 Scalar(a*Const(2) = 4.0 | Grad = 0.0),
 Scalar(a*Const(0.5) = 1.0 | Grad = 0.0),
 Scalar(*Const(2) = 1.0 | Grad = 0.0))`

##### Trancendental Functions
```python
a.log(), a.exp(), b.tanh()
```

`(Scalar(log(a) = 0.6931471805599453 | Grad = 0.0),
 Scalar(exp(a) = 7.38905609893065 | Grad = 0.0),
 Scalar(tanh(b) = 0.9950547536867305 | Grad = 0.0))`

## Directed Acyclic Graphs
A **Directed Graph** is created, when operations are performed.

```python
c = a + b.exp()
d = c * 3
e = d.log()
```

The `draw_dot(e)` function can be used on any `Scalar` of the computational graph for plotting the graph.

