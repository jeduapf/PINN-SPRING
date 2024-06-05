from ze import *
from tqdm.auto import tqdm, trange

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    set_cuda()
    torch.set_default_dtype(torch.float32)

    d, w0 = 2, 10
    system_params = {
                    "t": np.linspace(0,1,5000),
                    "m": 1,
                    "b": 2*d,
                    "k": w0**2,
                    "noise": 0.0,
                    "F0": 7*w0**2,
                    "W": 7*w0,
                    "x0": 1,
                    "x_0": 0}

    pinn_params = { "adaptative_start": 100.0, # FLOAT Adaptative proportion of loss funtion to which it will start learning parameters (Lo/Lf > adaptative_start)
                    "physics_points": 100,
                    "observation_points": 25,
                    "neurons": 80,
                    "layers": 3,
                    "stop_eps": 0.5*10**-2,
                    "stop_eps_u": 0.1*10**-2,
                    "learning_rate": 1*10**-3,
                    "regularization": 8*10**7,
                    "epochs": 50*10**3,
                    "batch": None,
                    "k_guess": 130.5,
                    "mu_guess": 8.3}

    if isinstance(pinn_params["adaptative_start"],float): 
        s = f'''Adaptatitve_start_b_{pinn_params['batch']}_n_{pinn_params['neurons']}_l_{pinn_params['layers']}_mu0_{pinn_params['mu_guess']:.1f}_k0_{pinn_params['k_guess']:.1f}_pys_{int(pinn_params['physics_points'])}_obs_{int(pinn_params['observation_points'])}_iter_{int(pinn_params['epochs']/1000)}k_lr_{pinn_params['learning_rate']:4.2e}_lb_{pinn_params['regularization']:4.2e}'''
    else: 
        s = f'''b_{pinn_params['batch']}_n_{pinn_params['neurons']}_l_{pinn_params['layers']}_mu0_{pinn_params['mu_guess']:.1f}_k0_{pinn_params['k_guess']:.1f}_pys_{int(pinn_params['physics_points'])}_obs_{int(pinn_params['observation_points'])}_iter_{int(pinn_params['epochs']/1000)}k_lr_{pinn_params['learning_rate']:4.2e}_lb_{pinn_params['regularization']:4.2e}'''

    SAVE_DIR, SAVE_GIF_DIR = DIRs(path_name = s, path_gif_name = 'gif')
    control_params = {  
                        'SEARCH' : False,
                        'LOSS_GIF': 500,
                        'PLOT': True,
                        'SAVE_DIR': SAVE_DIR,
                        'SAVE_GIF_DIR': SAVE_GIF_DIR,
                        'TEXT': True,
                        'FIGS': None}

    spring_pinn = PINN(control_params, system_params, pinn_params)
    spring_pinn.train()
    spring_pinn.save_plots()


    # b_None_n_80_l_3_mu0_8.9_k0_408.1_pys_100_obs_30_iter_10k_lr_1.00e-04_lb_5.00e+07