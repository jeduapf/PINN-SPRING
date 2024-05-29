
# Physics Informed Neural Network (PINN)
## Mass-spring-damper PDE

Exploring simple implementation of PINNs into a damped spring-mass system using pytorch as framework.

This code was largely based on Ben Moseley's workshop (https://github.com/benmoseley/harmonic-oscillator-pinn-workshop). He also launched a video in youtube explaining (https://www.youtube.com/watch?v=G_hIppUWcsc&t=3761s).

Here I'll explore a bit more this system including a different range of forced inputs as well as exploring the hyperparameters of the system with a simple grid search.


## Description 

Given a Spring-mass system following the equation:

$m\frac{\partial^2 x(t)}{\partial t^2} + b\frac{\partial x(t)}{\partial t} + kx(t) = 0
\\
m = 1\\
b = 2\xi\omega_0\\
k = w_0^2
\\
\frac{\partial^2 x(t)}{\partial t^2} + 2\xi\omega_0\frac{\partial x(t)}{\partial t} + w_0^2x(t) = 
0$

![Image](https://upload.wikimedia.org/wikipedia/commons/f/fa/Spring-mass_under-damped.gif)

Insira um gif ou um link de alguma demonstração


## Uso/Exemplos

```python
import ze
```

## References

<a id="1">[1]</a> 
Moseley, B (2022). 
Physics-informed machine learning: from concepts to real-world applications,
University of Oxford

<a id="2">[2]</a> 
Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E (2019). 
Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,
Elsevier

## Authors

- [@jeduapf](https://www.github.com/jeduapf)


## Tags

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

![Logo](https://www.univ-lyon1.fr/medias/photo/logolabo-ampere_1538049854649-jpg?ID_FICHE=1738)

