# Physics Informed Neural Network (PINN) - Hyper parameters tunning
## Mass-spring-damper PDE

The main focus of this branch is to visualize the losses and understand what are the impacts of different hyperparameters and frequencies.

## Results & Conclusion

To visualize the losses for the Spring PINN problem I separated in 3 types of visualization. The physics loss, the data loss and the meshgrid loss landscape calculated using the network parameters over a range of points of $b$ and $k$ in the inverse problem scénario and for the foward problem scénario.

### First the Data loss and Physics loss bellow for a 80 neurons 3 layers inverse problem where \mu_0 = 8.3 and k_0 = 378.5 with 100 physics points and 25 data points over 50000 iterations:

![Image](https://github.com/jeduapf/PINN-SPRING/blob/7d06f366115164d0146b5abec4a5332a5c378821/Adaptative_examples/b_None_n_80_l_3_mu0_8.3_k0_378.5_pys_100_obs_25_iter_50k_lr_5.00e-04_lb_8.00e%2B07/loss1.gif)
![Image](https://github.com/jeduapf/PINN-SPRING/blob/7d06f366115164d0146b5abec4a5332a5c378821/Adaptative_examples/b_None_n_80_l_3_mu0_8.3_k0_378.5_pys_100_obs_25_iter_50k_lr_5.00e-04_lb_8.00e%2B07/loss2.gif)

- For the loss landscape meshgrid: 
![Image](https://github.com/jeduapf/PINN-SPRING/blob/7d06f366115164d0146b5abec4a5332a5c378821/Adaptative_examples/b_None_n_80_l_3_mu0_8.3_k0_378.5_pys_100_obs_25_iter_50k_lr_5.00e-04_lb_8.00e%2B07/loss_3d.gif)

- In this exemple we can clearly see that the network learns pretty fast the data points "information" and struggles slowly to learn the physics "information" as the data loss converges to zero really fast while the physics loss keeps trying to adjust to the real (b,k) values of the equation.

### Now an failed convergency exemple for a 80 neurons 3 layers inverse problem where \mu_0 = 1.1 and k_0 = 1580.2 with 360 physics points and 90 data points over 500000 iterations:

![Image](https://github.com/jeduapf/PINN-SPRING/blob/9eeda1ec30008530c03cb1d059bc6eb9275c79cd/Adaptative_examples/b_None_n_80_l_3_mu0_1.1_k0_1580.2_pys_360_obs_90_iter_500k_lr_5.00e-05_lb_8.00e%2B07/loss1.gif)
![Image](https://github.com/jeduapf/PINN-SPRING/blob/9eeda1ec30008530c03cb1d059bc6eb9275c79cd/Adaptative_examples/b_None_n_80_l_3_mu0_1.1_k0_1580.2_pys_360_obs_90_iter_500k_lr_5.00e-05_lb_8.00e%2B07/loss2.gif)

- For the loss landscape meshgrid: 
![Image](https://github.com/jeduapf/PINN-SPRING/blob/9eeda1ec30008530c03cb1d059bc6eb9275c79cd/Adaptative_examples/b_None_n_80_l_3_mu0_1.1_k0_1580.2_pys_360_obs_90_iter_500k_lr_5.00e-05_lb_8.00e%2B07/loss_3d.gif)

- This is clearly a more difficult problem since the systems frequency changes from 20 rad/s to 40 rad/s. Even matching the Nyquist data and physics points and having way more iterations (10x more) the network can't converge for the same amount of parameters as before... Basiaclly doubling the systems frequency creates a flatter loss landscapeand the initial values of the inverse problem equation characteristics must be closer to the real value ones as shown in the Monte Carlo approach in another branch.

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

