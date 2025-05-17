# Physics Informed Neural Network (PINN) - Hyper parameters tunning
## Mass-spring-damper PDE

The main focus of this branch is to visualize the losses and understand what are the impacts of different hyperparameters and frequencies.

## Results & Conclusion

To visualize the losses for the Spring PINN problem I separated in 3 types of visualization. The physics loss, the data loss and the meshgrid loss landscape calculated using the network parameters over a range of points of $b$ and $k$ in the inverse problem scénario and for the foward problem scénario.

- First the Data loss and Physics loss bellow for a 80 neurons 3 layers inverse problem where \mu_0 = 8.3 and k_0 = 378.5 with 100 physics points and 25 data points over 50000 iterations:

![Image](https://github.com/jeduapf/PINN-SPRING/blob/7d06f366115164d0146b5abec4a5332a5c378821/Adaptative_examples/b_None_n_80_l_3_mu0_8.3_k0_378.5_pys_100_obs_25_iter_50k_lr_5.00e-04_lb_8.00e%2B07/loss1.gif)
![Image](https://github.com/jeduapf/PINN-SPRING/blob/7d06f366115164d0146b5abec4a5332a5c378821/Adaptative_examples/b_None_n_80_l_3_mu0_8.3_k0_378.5_pys_100_obs_25_iter_50k_lr_5.00e-04_lb_8.00e%2B07/loss2.gif)

- For the loss landscape meshgrid: 
![Image](https://github.com/jeduapf/PINN-SPRING/blob/7d06f366115164d0146b5abec4a5332a5c378821/Adaptative_examples/b_None_n_80_l_3_mu0_8.3_k0_378.5_pys_100_obs_25_iter_50k_lr_5.00e-04_lb_8.00e%2B07/loss_3d.gif)


## Different system dynamics

Here the main objective is to test if the initial guess for b an k have significant impact in the convergency of the PINN. For each system dynamics a threshold of 1% for b, k and the predicted function $x(t)$ is set and if the training achieves the three thresholds simultaniously the training is finished. 

A maximum of $200 000$ epochs and learning rate of $10^{-4}$ is used for training as well as a regularization term of $3000000$ betweend data and physics losses. 

The neural network is a fully connected with 80 neurons per layer and 3 layers.

A total of $1000$ physics grid points were taken as well as $200$ data points. 

Initial guesses are taken from a multivariate gaussin distribution with 2 times the variance not exceeding the maximum theorical difference from the ground truth value and initial guess distance ($distance_{max} = learning \ rate \cdot epochs$)

The maximum frequency of the system is its natural frequency given by $f_0 = \frac{w_0}{2\pi}$ and $w_0 = \sqrt{k}$

### Results & Conclusion

$$w_0 = 20 \ rad/s$$

![Image](https://github.com/jeduapf/PINN-SPRING/blob/hypertuning/monte_carlo/monte_carlo_b_4.00_k_400.00_harmonic_3.png?raw=true)

$$w_0 = 40 \ rad/s$$

![Image](https://github.com/jeduapf/PINN-SPRING/blob/hypertuning/monte_carlo/monte_carlo_b_4.00_k_1600.00_harmonic_3.png?raw=true)

$$w_0 = 80 \ rad/s$$

![Image](https://github.com/jeduapf/PINN-SPRING/blob/hypertuning/monte_carlo/monte_carlo_b_4.00_k_6400.00_harmonic_3.png?raw=true)

Although the testing framework is still statistical small one can intuitively conclude that the higher the frequency (for a fixed set of hyperparameters) more difficult it will be for the network to converge and find the good constants of the PDE.

Also from the low frequency problems, most of the time it converged between $20 000$ to $60 000$ iterations and for higher frequencies it didn't not only not converged but also went through all the $200 000$ epochs...

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

