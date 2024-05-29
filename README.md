# Physics Informed Neural Network (PINN)
## Mass-spring-damper PDE

With the same assumptions from before now the obejetive is to insert a oscilating force into the system and raise its maximum frequency.

In this project the **main objective** is to discover the parameters of the equation described bellow (b and k, assuming m = 1 Kg).

## Description 

Given a Spring-mass homogeneous system following the equation ,if we assume m = 1 Kg and b and k as the following:

$$\frac{\partial^2 x(t)}{\partial t^2} + b\frac{\partial x(t)}{\partial t} + kx(t) = 0$$

Adding a external force that is a function of time the equation becomes:

$$\frac{\partial^2 x(t)}{\partial t^2} + b\frac{\partial x(t)}{\partial t} + kx(t) = F_{ext}(t)$$


