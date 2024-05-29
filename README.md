

# Physics Informed Neural Network (PINN)
## Mass-spring-damper PDE

Exploring simple implementation of PINNs into a damped spring-mass system using pytorch as framework.

This code was largely based on Ben Moseley's workshop (https://github.com/benmoseley/harmonic-oscillator-pinn-workshop). He also launched a video in youtube explaining (https://www.youtube.com/watch?v=G_hIppUWcsc&t=3761s).

Here I'll explore a bit more this system including a different range of forced inputs as well as exploring the hyperparameters of the system with a simple grid search.

In this project the <b>main objective<b> is to discover the parameters of the equation described bellow (b and k, assuming m = 1 Kg).


## Description 

Given a Spring-mass system following the equation:

$ m\frac{\partial^2 x(t)}{\partial t^2} + b\frac{\partial x(t)}{\partial t} + kx(t) = 0 $


