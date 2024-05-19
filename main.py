from ze import *

def hyperparameter_search(layers, neurons, learning_rate):
    
    for lr in learning_rate:
            for la in layers:
                for ne in neurons:
                    # s = f'''b_{pinn_params['batch']}_n_{pinn_params['neurons']}_l_{pinn_params['layers']}_mu0_{pinn_params['mu_guess']:.1f}_k0_{pinn_params['k_guess']:.1f}_pys_{int(pinn_params['physics_points'])}_obs_{int(pinn_params['observation_points'])}_iter_{int(pinn_params['epochs']/1000)}k_lr_{pinn_params['learning_rate']:4.2e}_lb_{pinn_params['regularization']:4.2e}'''
                    s = "hyperparameter_search"
                    SAVE_DIR, SAVE_GIF_DIR = DIRs(path_name = s, path_gif_name = 'gif')
                    control_params = {  'SEARCH': True,
                                        'PLOT': False,
                                        'SAVE_DIR': SAVE_DIR,
                                        'SAVE_GIF_DIR': SAVE_GIF_DIR,
                                        'TEXT': False,
                                        'FIGS': None}
                    d, w0 = 2, 20
                    system_params = {
                                    "t": np.linspace(0,1,5000),
                                    "m": 1,
                                    "b": 2*d,
                                    "k": w0**2,
                                    "noise": 0.0,
                                    "F0": 300,
                                    "W": 3*w0,
                                    "x0": 1,
                                    "x_0": 0}

                    pinn_params = {
                        "physics_points": 300, # Maximum frequency of the signal possible f = 150 Hz = 940 rad/s ( better choose freq_max as f_nyquist/4)
                        "observation_points": 50,
                        "neurons": ne,
                        "layers": la,
                        "learning_rate": lr,
                        "regularization": 5*10**5,
                        "epochs": 10**3,
                        "batch": None,
                        "k_guess": 406.1,
                        "mu_guess": -0.3}


                    # Change the force a bit too
                    system_params["F0"] = np.random.normal(system_params["F0"], system_params["k"]/4, 1)[0] 
                    system_params["W"] = np.random.normal(system_params["W"], np.sqrt(system_params["k"]), 1)[0] 
                    if system_params["F0"] < 100: 
                        system_params["F0"] = 300
                    if system_params["W"] < 0: 
                        system_params["W"] = 10

                    # Randomly select the initial guesses in the normal distribution centered in the true value
                    # with 2 standard deviation of the distance of the total steps times the learning rate ( rough approximation ) 
                    pinn_params["mu_guess"] = 2*d + np.random.normal(0, pinn_params["epochs"]*lr/2, 1)[0]
                    pinn_params["k_guess"] = w0**2 + np.random.normal(0, pinn_params["epochs"]*lr/2, 1)[0]

                    spring_pinn = PINN(control_params, system_params, pinn_params)
                    spring_pinn.train()
                    spring_pinn.save_results()

def run_once():
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
                    "epochs": 5*10**4,
                    "batch": None,
                    "k_guess": 406.1,
                    "mu_guess": -0.3}

    s = f'''b_{pinn_params['batch']}_n_{pinn_params['neurons']}_l_{pinn_params['layers']}_mu0_{pinn_params['mu_guess']:.1f}_k0_{pinn_params['k_guess']:.1f}_pys_{int(pinn_params['physics_points'])}_obs_{int(pinn_params['observation_points'])}_iter_{int(pinn_params['epochs']/1000)}k_lr_{pinn_params['learning_rate']:4.2e}_lb_{pinn_params['regularization']:4.2e}'''
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

if __name__ == "__main__":
    set_cuda()
    torch.set_default_dtype(torch.float32)
    

    layers = [3]
    neurons = [80]
    learning_rate = [1*10**-3, 1*10**-4]
    hyperparameter_search(layers, neurons, learning_rate)