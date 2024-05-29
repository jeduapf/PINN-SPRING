# Physics Informed Neural Network (PINN) - Hyper parameters tunning
## Mass-spring-damper PDE

The main focus of this branch is to find the 'optimal' parameters for the problem (if exist). Basically doing several simulations for each different tiplet of hyperparameters (Neurons, Layers, Learning Rate) to the **SAME** system dynamics and verifying which triplet solves better the problem and which variable influences the most.

This idea was largely based on  He-Wen-Xuan Li's paper: Motion estimation and system identification of amoored buoy via physics-informed neural network ([article](https://www.sciencedirect.com/science/article/abs/pii/S0141118723002183)).

## Description 

Given the same Spring-mass system following the equation:

$$\frac{\partial^2 x(t)}{\partial t^2} + 2\xi\omega_0\frac{\partial x(t)}{\partial t} + w_0^2x(t) = 0$$

Assume the underlying solution $x(t)$ of the equation as a Fully Connected Neural Network where its enteries are time points

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

Train the network for each combination of the hyperparameters:

$$(N,L,Lr)$$ such that $N \in [40,80,120]$, $L \in [3,6,9]$, $N \in [5 \cdot 10^-2,\ 5 \cdot 10^-3,\ 10^-3,\ 10^-4,\ 10^-5,\ 5 \cdot 10^-6]$

For a fixed amount of epochs and initial k and b such that the steps needed to achive the solution is at least $0.5 * epochs * LearningRate$

## Conclusion

![Image](https://raw.githubusercontent.com/jeduapf/PINN-SPRING/hypertuning/hyperparameter_search/errors.png)

In this experiment the strogenst conclusion is that a bad choice of Learning Rate implies in bad estimation and the architecture of the network doesn't change much its convergency (except for the time to train which isn't shown above). 

Also, this experiment was on a **SINGLE** case problem with low frequency as shown below:

![Image](https://raw.githubusercontent.com/jeduapf/PINN-SPRING/main/Converged/mu0_13.0_k0_417.0_pys_300_obs_60_iter_100k_lr_3.00e-04_lb_1.00e%2B05/learning_k_mu.gif)

![Image](https://github.com/jeduapf/PINN-SPRING/blob/main/Converged/mu0_13.0_k0_417.0_pys_300_obs_60_iter_100k_lr_3.00e-04_lb_1.00e+05/loss1.gif?raw=true)

## References

[[1]](https://ora.ox.ac.uk/objects/uuid:b790477c-771f-4926-99c6-d2f9d248cb23/files/d8p58pd35h)
Moseley, B (2022). 
Physics-informed machine learning: from concepts to real-world applications,
University of Oxford

[[2]](https://arxiv.org/pdf/1711.10561)
Maziar Raissi, Paris Perdikaris, and George Em Karniadakis (2017). 
Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations

## Authors

- [@jeduapf](https://www.github.com/jeduapf)


## Tags

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

![Logo](https://www.univ-lyon1.fr/medias/photo/logolabo-ampere_1538049854649-jpg?ID_FICHE=1738)

