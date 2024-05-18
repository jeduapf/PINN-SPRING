from ze import *

if __name__ == "__main__":
    set_cuda()
    torch.set_default_dtype(torch.float32)
        
    d, w0 = 2, 20
    system_params = {
                    "t": np.linspace(0,1,5000),
                    "m": 1,
                    "b": 2*d,
                    "k": w0**2,
                    "noise": 0.0,
                    "F0": 300,
                    "W": 7*w0,
                    "x0": 1,
                    "x_0": 0}

    pinn_params = {
                    "physics_points": 300,
                    "observation_points": 50,
                    "neurons": 80,
                    "layers": 3,
                    "learning_rate": 1*10**-3,
                    "regularization": 5*10**5,
                    "epochs": 6*10**4,
                    "batch": None,
                    "k_guess": 406.1,
                    "mu_guess": -0.3}

    s = f'''n_{pinn_params['neurons']}_l_{pinn_params['layers']}_mu0_{pinn_params['mu_guess']:.1f}_k0_{pinn_params['k_guess']:.1f}_pys_{int(pinn_params['physics_points'])}_obs_{int(pinn_params['observation_points'])}_iter_{int(pinn_params['epochs']/1000)}k_lr_{pinn_params['learning_rate']:4.2e}_lb_{pinn_params['regularization']:4.2e}'''
    SAVE_DIR, SAVE_GIF_DIR = DIRs(path_name = s, path_gif_name = 'gif')
    control_params = {
                        'PLOT': True,
                        'SAVE_DIR': SAVE_DIR,
                        'SAVE_GIF_DIR': SAVE_GIF_DIR,
                        'TEXT': True,
                        'FIGS': None}

    spring_pinn = PINN(control_params, system_params, pinn_params)
    spring_pinn.train()
    spring_pinn.save_plots()

