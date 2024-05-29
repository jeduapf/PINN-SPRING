# Physics Informed Neural Network (PINN)
## Mass-spring-damper PDE

With the same assumptions from before now the obejetive is to insert a oscilating force into the system and raise its maximum frequency.

In this project the **main objective** is to discover the parameters of the equation described bellow (b and k, assuming m = 1 Kg).

## Description 

Given a Spring-mass homogeneous system following the equation ,if we assume m = 1 Kg and b and k as the following:

$$\frac{\partial^2 x(t)}{\partial t^2} + b\frac{\partial x(t)}{\partial t} + kx(t) = 0$$

Adding a external force that is a function of time the equation becomes:

$$\frac{\partial^2 x(t)}{\partial t^2} + b\frac{\partial x(t)}{\partial t} + kx(t) = F_{ext}(t)$$

## Results

### Converged

- System 
	- $b = 4$ 
	- $k = 400$ 
	- $w_0 = 20$

- Force: 
	- $A = 300 N$ 
	- $W = 140 rad/s = 7 \cdot w_0$

- Guesses:
	- $b = 7$
	- $k = 390$

- Points:
	- Physics: 120
	- Data: 20 

- Parameters:
	- Learning rate: $5\cdot10^{-5}$
	- Epochs: 500 000
	- Neurons: 80
	- Layers: 3 
	- Lambda: $10^5$


![Image](https://github.com/jeduapf/PINN-SPRING/blob/external_force/Converged/BON-mu0_7.0_k0_390.0_pys_120_obs_20_iter_500k_lr_2.00e-05_lb_5.00e+05/learning_k_mu.gif?raw=true)

![Image](https://github.com/jeduapf/PINN-SPRING/blob/external_force/Converged/BON-mu0_7.0_k0_390.0_pys_120_obs_20_iter_500k_lr_2.00e-05_lb_5.00e+05/loss1.gif?raw=true)

![Image](https://github.com/jeduapf/PINN-SPRING/blob/external_force/Converged/BON-mu0_7.0_k0_390.0_pys_120_obs_20_iter_500k_lr_2.00e-05_lb_5.00e+05/Constants.png?raw=true)

![Image](https://github.com/jeduapf/PINN-SPRING/blob/external_force/Converged/BON-mu0_7.0_k0_390.0_pys_120_obs_20_iter_500k_lr_2.00e-05_lb_5.00e+05/Loss.png?raw=true)


### Didn't converge (perfectly)

- System 
	- $b = 4$ 
	- $k = 1600$ 
	- $w_0 = 40$

- Force: 
	- $A = 1000 N$ 
	- $W = 40 rad/s$

- Guesses:
	- $b = -5.9$
	- $k = 1594.9$

- Points:
	- Physics: 5400
	- Data: 180 

- Parameters:
	- Learning rate: $5\cdot10^{-5}$
	- Epochs: 100 000
	- Neurons: 80
	- Layers: 3 
	- Lambda: $5.84\cdot10^7$

![Image](https://github.com/jeduapf/PINN-SPRING/blob/external_force/More%20or%20less/mu0_-5.9_k0_1594.9_pys_5400_obs_180_iter_100k_lr_1.00e-04_lb_5.84e+07/learning_k_mu.gif?raw=true)

![Image](https://github.com/jeduapf/PINN-SPRING/blob/external_force/More%20or%20less/mu0_-5.9_k0_1594.9_pys_5400_obs_180_iter_100k_lr_1.00e-04_lb_5.84e+07/loss1.gif?raw=true)

![Image](https://github.com/jeduapf/PINN-SPRING/blob/external_force/More%20or%20less/mu0_-5.9_k0_1594.9_pys_5400_obs_180_iter_100k_lr_1.00e-04_lb_5.84e+07/Constants.png?raw=true)

![Image](https://github.com/jeduapf/PINN-SPRING/blob/external_force/More%20or%20less/mu0_-5.9_k0_1594.9_pys_5400_obs_180_iter_100k_lr_1.00e-04_lb_5.84e+07/Loss.png?raw=true)
