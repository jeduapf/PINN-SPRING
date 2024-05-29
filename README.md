# Physics Informed Neural Network (PINN)
## Mass-spring-damper PDE

Exploring simple implementation of PINNs into a damped spring-mass system using pytorch as framework.

This code was largely based on Ben Moseley's workshop [benmoseley](https://github.com/benmoseley/harmonic-oscillator-pinn-workshop). He also launched a video in youtube explaining [youtube](https://www.youtube.com/watch?v=G_hIppUWcsc&t=3761s).

Here I'll explore a bit more this system including a different range of forced inputs as well as exploring the hyperparameters of the system with a simple grid search.

In this project the **main objective** is to discover the parameters of the equation described bellow (b and k, assuming m = 1 Kg).


## Description 

Given a Spring-mass system following the equation:

$m\frac{\partial^2 x(t)}{\partial t^2} + b\frac{\partial x(t)}{\partial t} + kx(t) = 0$

If we assume m = 1 Kg and b and k as the following:
$$m = 1 $$ 
$$b = 2\xi\omega_0 (b=\mu \\ in \\ the \\ code)$$ 
$$k = w_0^2$$

We get the homogeneous equation bellow:
$\frac{\partial^2 x(t)}{\partial t^2} + 2\xi\omega_0\frac{\partial x(t)}{\partial t} + w_0^2x(t) = 
0$

Litte gif showing the form of the position $x(t)$ of the spring and its derivative $\frac{\partial x(t)}{\partial t}$ over time for the homogeneous equation above:

![Image](https://upload.wikimedia.org/wikipedia/commons/f/fa/Spring-mass_under-damped.gif)

Now let's assume we want to approximate the underlying solution $x(t)$ of the equation as a Fully Connected Neural Network where its enteries are time points

```math
t = \begin{pmatrix}t_0\\\vdots\\t_k\end{pmatrix}
```
and its outputs are the values of $x(t)$

```math
x(t) = \begin{pmatrix}x(t_0)\\\vdots\\x(t_k)\end{pmatrix}
```
for any given t in the domain $t \in D = [a,b]$. So $NN(t) \approx x(t), \forall t \in D$.

Also let's add the parameters we want to discover as "weights" in the optimization of the Neural Network. So b and k are also added as learnable parameters in the training.

```python
self.k_guess = torch.nn.Parameter(torch.tensor([float(pinn_params["k_guess"])], requires_grad=True))
self.mu_guess = torch.nn.Parameter(torch.tensor([float(pinn_params["mu_guess"])], requires_grad=True))

self.optimiser = torch.optim.Adam(list(self.pinn.parameters())+[self.k_guess, self.mu_guess],lr=self.learning_rate, betas=(0.95, 0.999))
```


## References

[1](https://ora.ox.ac.uk/objects/uuid:b790477c-771f-4926-99c6-d2f9d248cb23/files/d8p58pd35h)
Moseley, B (2022). 
Physics-informed machine learning: from concepts to real-world applications,
University of Oxford

[2](https://arxiv.org/pdf/1711.10561)
Maziar Raissi, Paris Perdikaris, and George Em Karniadakis (2017). 
Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations

## Authors

- [@jeduapf](https://www.github.com/jeduapf)


## Tags

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

![Logo](https://www.univ-lyon1.fr/medias/photo/logolabo-ampere_1538049854649-jpg?ID_FICHE=1738)

