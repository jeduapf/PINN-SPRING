from tqdm.auto import tqdm, trange
import numpy as np
import torch
from ze.utils import *
from ze.torch_utils import FCN
import csv
import os
import json

class PINN():

    def __init__(self, control_params, system_params, pinn_params):

        # Dashboard parameters
        self.PLOT = control_params['PLOT']
        self.SAVE_DIR = control_params['SAVE_DIR']
        self.SAVE_GIF_DIR = control_params['SAVE_GIF_DIR']
        self.TEXT = control_params['TEXT']
        self.FIGS = control_params['FIGS']
        self.SEARCH = control_params['SEARCH']
        self.LOSS_GIF = control_params['LOSS_GIF']

        # ---------------------------------------- System ----------------------------------------
        # System parameters
        self.sys_params = system_params
        self.t = system_params["t"] # Linspace of time points (Maximum amount of points)
        self.m = system_params["m"]
        self.b = system_params["b"]
        self.k = system_params["k"]
        self.noise = system_params["noise"]
        self.d = self.b/2
        self.w0 = np.sqrt(self.k)
        self.k_torch = torch.tensor([float(self.k)], dtype = torch.float32)
        self.b_torch = torch.tensor([float(self.b)], dtype = torch.float32)

        # Forced input parameters
        self.F0 = system_params["F0"]
        self.W = system_params["W"]
        self.x0 = system_params["x0"]
        self.x_0 = system_params["x_0"]

        # ---------------------------------------- PINN ----------------------------------------
        # PINN construction parameters
        self.physics_points = pinn_params["physics_points"]
        self.adaptative_start = pinn_params["adaptative_start"]
        self.observation_points = pinn_params["observation_points"]
        self.neurons = pinn_params["neurons"]
        self.layers = pinn_params["layers"]
        assert isinstance(self.physics_points,int) and isinstance(self.observation_points,int) and (self.physics_points<len(self.t) and self.observation_points<len(self.t)), "Amount of data and physics points must be an integer smaller than the total amout of points in the simulation !"
        
        # PINN neural network
        self.pinn = FCN(1,1,self.neurons,self.layers)

        # PINN optimization parameters
        self.stop_eps = pinn_params["stop_eps"]
        self.stop_eps_u = pinn_params["stop_eps_u"]
        self.learning_rate = pinn_params["learning_rate"]
        self.regularization = pinn_params["regularization"]
        self.epochs = pinn_params["epochs"]
        self.batch = pinn_params["batch"]

        if isinstance(self.adaptative_start,float):
            self.k_guess = torch.nn.Parameter(torch.tensor([float(pinn_params["k_guess"])], requires_grad=True)).requires_grad_(False)
            self.mu_guess = torch.nn.Parameter(torch.tensor([float(pinn_params["mu_guess"])], requires_grad=True)).requires_grad_(False)
        else:
            self.k_guess = torch.nn.Parameter(torch.tensor([float(pinn_params["k_guess"])], requires_grad=True))
            self.mu_guess = torch.nn.Parameter(torch.tensor([float(pinn_params["mu_guess"])], requires_grad=True))

        # optimizer
        self.optimiser = torch.optim.Adam(list(self.pinn.parameters())+[self.k_guess, self.mu_guess],lr=self.learning_rate, betas=(0.95, 0.999))

        # ---------------------------------------- POINTS ----------------------------------------
        # Observation vectors ( Don't require grad )
        self.t_obs_np = np.random.choice(self.t, self.observation_points, replace=False) 
        self.t_obs = torch.tensor(self.t_obs_np, dtype=torch.float32).view(-1,1)
                                                                        # Change randn if it isn't just between [0-1]
        self.u_obs_np = forced_damped_spring(self.t_obs_np, system_params) #+ self.noise*np.random.randn(self.t_obs_np.shape[0],self.t_obs_np.shape[1])
        self.u_obs = torch.tensor(self.u_obs_np, dtype=torch.float32).view(-1,1)

        # Physics vectors ( Require grad )
        self.t_physics = torch.linspace(0,1,self.physics_points, dtype=torch.float32).view(-1,1).requires_grad_(True)
        self.t_physics_np = self.t_physics.detach().cpu().numpy()

        # Force vectors ( Don't require grad )
        self.force = torch.tensor( self.F0 * np.cos(self.W*self.t_physics_np)/self.m, dtype=torch.float32).view(-1,1)
        self.force_np = self.force.detach().cpu().numpy().squeeze()

        # Validation/test 
        self.t_test = torch.linspace(0,1,1000, dtype=torch.float32).view(-1,1)
        self.t_test_np = self.t_test.detach().cpu().numpy()
        self.u_exact = torch.tensor(forced_damped_spring(self.t_test_np, system_params), dtype=torch.float32).view(-1,1)

        # ---------------------------------------- Plotting ----------------------------------------
        # Tracking
        self.constants = []
        self.losses = []
        self.derivatives = []
        self.files = []

        if self.PLOT:
            plot_initial(   self.t_obs_np.squeeze(), 
                            self.u_obs_np.squeeze(), 
                            self.t_physics_np.squeeze(), 
                            system_params,
                            self.SAVE_DIR)

            # define grid in case of loss gif
            if isinstance(self.LOSS_GIF,int):
                self.ks = np.linspace(0.9*self.k, 1.1*self.k, self.LOSS_GIF)
                self.bs = np.linspace(-0.5*self.b,  2.5*self.b, self.LOSS_GIF) 
                self.Bs, self.Ks = np.meshgrid(self.bs, self.ks)
                
                self.SAVE_DIR_LOSS = os.path.join(self.SAVE_DIR,"loss_gif")
                try:
                    os.mkdir(self.SAVE_DIR_LOSS)
                except:
                    pass

                self.files_loss = []

    def physics_loss_surface(self, iteration):
        X = self.Bs
        Y = self.Ks
        Z = np.zeros(X.shape)

        # # Position of the minimum loss of each axis
        # J = np.argmin( np.abs(X[0,:] - self.constants[-1][0]*np.ones(X[0,:].shape)) )
        # I = np.argmin( np.abs(Y[:,0] - self.constants[-1][1]*np.ones(Y[:,0].shape)) )

        # Calculate all physics loss values for each point combination in the grid
        for i in range(self.LOSS_GIF):
            for j in range(self.LOSS_GIF):
                Z[i][j] = self.physics_loss_np(X[i][j], Y[i][j])

        fig = plt.figure(figsize = (17,15))
        ax = fig.add_subplot(111, projection='3d')

        # 3D surface of the chosen meshgrid
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        
        # Points on the ground (current point and expected)
        ax.scatter(self.b, self.k, color = 'green', s = 30, label = 'Ground Truth')
        ax.scatter(self.constants[-1][0], self.constants[-1][1], color='#ff5700', s=40, label = 'Current predicted value') # Add the current point of the network

        # Dashed lines to see the current point position on the ground
        ax.plot([self.constants[-1][0], self.constants[-1][0]], [np.min(Y), np.max(Y)], color='#ff5700', linewidth=3, linestyle='dashed')
        ax.plot([np.min(X), np.max(X)], [self.constants[-1][1], self.constants[-1][1]], color='#ff5700', linewidth=3, linestyle='dashed')

        ax.set_title(f'Loss of {self.LOSS_GIF} mesh grid points at iteration {iteration}\n b={self.constants[-1][0]:.3f}, k={self.constants[-1][1]:.3f}', fontsize=30)
        ax.set_xlabel('b')
        ax.set_ylabel('K')
        ax.set_zlabel('||f||')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Remove values of X and Y axis
        ax.set_xticks([])
        ax.set_yticks([])

        # Save figure for GIF and track path
        file_name = os.path.join(self.SAVE_DIR_LOSS,"3d_loss_%.8i.png"%(iteration+1))
        plt.savefig(file_name, dpi=100, facecolor="white")
        self.files_loss.append(file_name)
        plt.close(fig)

    def physics_loss_np(self, b, k):

        d2udt2 = self.derivatives[-1][0]
        dudt = self.derivatives[-1][1]
        u_phy_hat = self.derivatives[-1][2]
        f = self.force_np
        
        return np.mean((d2udt2 + b*dudt + k*u_phy_hat - f)**2) 

    def physics_loss(self, t_physics, force):
        # PHYSICS LOSS
        u_phy_hat = self.pinn(t_physics)
        dudt = torch.autograd.grad( u_phy_hat, t_physics, 
                                    torch.ones_like(u_phy_hat), 
                                    create_graph=True)[0]

        d2udt2 = torch.autograd.grad(   dudt, t_physics, 
                                        torch.ones_like(dudt), 
                                        create_graph=True)[0]


        self.derivatives.append([d2udt2.detach().cpu().numpy().squeeze(),
                                dudt.detach().cpu().numpy().squeeze(),
                                u_phy_hat.detach().cpu().numpy().squeeze()])

        self.constants.append([ self.mu_guess.item(),
                                self.k_guess.item()])

        return torch.mean((d2udt2 + self.mu_guess*dudt + self.k_guess*u_phy_hat - force)**2) # Now with the imposed force term

    def data_loss(self, t_obs, u_obs):
        # DATA LOSS
        u_obs_hat =  self.pinn(t_obs)
        self.derivatives[-1].append(u_obs_hat.detach().cpu().numpy().squeeze())

        return torch.mean((u_obs_hat - u_obs)**2) 

    def stop(self):
        return self.losses[-1][2] < self.stop_eps_u*self.losses[0][2] and torch.abs(self.constants[-1][0]/self.b_torch - 1) < self.stop_eps and torch.abs(self.constants[-1][1]/self.k_torch - 1) < self.stop_eps

    def dashboard(self, iteration):
        if self.TEXT:
            tqdm.write(f"{iteration}\n>>Physics: {self.losses[-1][0]:.3f} >>Data: {self.losses[-1][1]:.3f} >>Total: {self.losses[-1][2]:.3f}\n>>Mu: {self.constants[-1][0]:.3f} >>k: {self.constants[-1][1]:.3f}\n")

        if self.PLOT:

            fig = plt.figure(figsize=(12,5))

            # To not compute any gradients in this phase
            with torch.no_grad():
                self.pinn.eval()  # Evaluation mode
                u = self.pinn(self.t_test)
            

                plt.scatter(self.t_obs_np, self.u_obs_np, label="Noisy observations", alpha=0.6)
                plt.plot(self.t_test_np, u.detach().cpu().numpy(), label="PINN solution", color="tab:green")
                plt.title(f"Training step {iteration}")
                plt.legend()

                file = os.path.join(self.SAVE_GIF_DIR,"pinn_%.8i.png"%(iteration+1))
                plt.savefig(file, dpi=100, facecolor="white")
                self.files.append(file)
                plt.close(fig)
                
                if isinstance(self.LOSS_GIF,int):
                    self.physics_loss_surface(iteration)

            self.pinn.train(True) # Back to trainning mode  
                
    def step(self, i, t_physics, force):

        self.optimiser.zero_grad()

        phy_loss = self.physics_loss(t_physics, force)
        dat_loss = self.data_loss(self.t_obs, self.u_obs)
        loss = phy_loss + self.regularization*dat_loss

        loss.backward()

        self.optimiser.step()

        self.losses.append([phy_loss.item(),
                            dat_loss.item(),
                            loss.item()])

    def train(self, LOSS_GIF = False):
        self.pinn.train() # Set model to trainning mode

        if self.FIGS is None:
            self.FIGS = int(self.epochs/100)

        if self.batch is None:
            bar = trange(self.epochs)
            for i in bar:

                if isinstance(self.adaptative_start,float):
                    if i > 10 and self.losses[0][-1] > self.adaptative_start*self.losses[-1][-1]:
                        self.k_guess.requires_grad_(True)
                        self.mu_guess.requires_grad_(True)


                self.step(i, self.t_physics, self.force)

                # plot the result as training progresses
                if i % self.FIGS == 0:
                    self.dashboard(i)

                # Early stopping in case of convergency
                if self.stop():
                    print("\n\n\t\t Converged, finishing early !\n\n")
                    break

        # ONLY APPLY BATCH TO PHYSICS POINTS 
        else:
            assert isinstance(self.batch,int) and self.batch > 0 and self.batch < self.physics_points, "Batch size must be a positive integer smaller than the number of physics points..."
            p = np.linspace(0, self.physics_points - 1, self.physics_points, dtype = int)

            bar = trange(self.epochs)
            for i in bar:

                for _i in range(0, self.physics_points, self.batch):
                    indices = np.random.choice(p, size=self.batch, replace=False)
                    self.step(i, self.t_physics[indices], self.force[indices])

                # plot the result as training progresses
                if i % self.FIGS == 0:
                    self.dashboard(i)

                # Early stopping in case of convergency
                if self.stop():
                    print("\n\n\t\t Converged, finishing early !\n\n")
                    break

    def save_plots(self):

        if self.batch is not None:
            p = np.linspace(0, self.physics_points - 1, self.batch, dtype = int)
            force = self.force_np[p]
        else:
             force = self.force_np

        files1, files2 = write_losses(  self.u_obs, self.derivatives, self.constants, 
                                        self.SAVE_DIR, self.losses, force, 
                                        l = self.regularization, TEXT = False, PLOT = True, 
                                        fig_pass = self.FIGS, SAVE_PATH = self.SAVE_DIR)

        losses_constants_plot(self.constants, self.losses, self.SAVE_DIR, self.d, self.w0)

        print("\n\nGenerating GIFs...\n\n")
        save_gif_PIL(os.path.join(self.SAVE_DIR,"learning_k_mu.gif"), self.files, fps=60, loop=0)
        save_gif_PIL(os.path.join(self.SAVE_DIR,"loss1.gif"), files1, fps=60, loop=0)
        save_gif_PIL(os.path.join(self.SAVE_DIR,"loss2.gif"), files2, fps=60, loop=0)
        save_gif_PIL(os.path.join(self.SAVE_DIR,"loss_3d.gif"), self.files_loss, fps=60, loop=0)

    def predict(self, t):
        with torch.no_grad():
            self.pinn.eval()
            return self.pinn(t).detach().cpu().numpy().squeeze()

    def save_results(self):
        arr_loss = np.array(self.losses)
        u_hat = self.predict(torch.tensor(self.t, dtype=torch.float32).view(-1,1))
        u_star = forced_damped_spring(self.t, self.sys_params)

        fields=[self.learning_rate,
                arr_loss.shape[0],
                self.layers, 
                self.neurons, 
                np.abs(self.k_guess.item()/self.k - 1),
                np.abs(self.mu_guess.item()/self.b - 1),
                np.sqrt( np.mean( np.abs( u_hat - u_star)**2 ) ),
                u_hat,
                u_star,
                arr_loss[:,0],
                arr_loss[:,1],
                arr_loss[:,2]]

        print()
        print(f"Total iterations: {int(arr_loss.shape[0])}")
        print(f"K value from pinn: {self.k_guess.item():.3f}")
        print(f"K relative error from pinn: {np.abs((self.k_guess.item()/self.k - 1))*100:.3f}%")
        print(f"Mu value from pinn: {self.mu_guess.item():.3f}")
        print(f"Mu relative error from pinn: {np.abs((self.mu_guess.item()/self.b - 1))*100:.3f}%")
        print(f"Mean error of all {int(len(u_star))} points in function: {np.sqrt( np.mean( np.abs( u_hat - u_star)**2 ) ):.4f}")
        print()

        with open(os.path.join(self.SAVE_DIR,'results.csv'), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(fields)

    # TODO: check if JSON is saving and READING the way I want
    def save_monte_carlo(self, pinn_params):
        u_hat = self.predict(torch.tensor(self.t, dtype=torch.float32).view(-1,1))
        u_star = forced_damped_spring(self.t, self.sys_params)
        consts = np.array(self.constants)

        fields={"val_error": np.sqrt( np.mean( np.abs( u_hat - u_star)**2 ) ),
                "true_b":self.b, 
                "true_k":self.k,
                "pinn_b":list(consts[:,0]),
                "pinn_k":list(consts[:,1])}
        print()
        print(f"Total iterations: {int(consts.shape[0])}")
        print(f"K value from pinn: {self.k_guess.item():.3f}")
        print(f"K relative error from pinn: {np.abs((self.k_guess.item()/self.k - 1))*100:.3f}%")
        print(f"Mu value from pinn: {self.mu_guess.item():.3f}")
        print(f"Mu relative error from pinn: {np.abs((self.mu_guess.item()/self.b - 1))*100:.3f}%")
        print(f"Mean error of all {int(len(u_star))} points in function: {np.sqrt( np.mean( np.abs( u_hat - u_star)**2 ) ):.4f}")
        print()

        with open(os.path.join(self.SAVE_DIR,f'monte_carlo_b_{self.b:.2f}_k_{self.k:.2f}_harmonic_{int(self.W/self.w0)}.json'), 'a') as f:
            f.write(str(fields).replace("'",'"') + '\n')

        with open(os.path.join(self.SAVE_DIR,f'pinn_params.txt'), 'w') as f:
            f.write(str(pinn_params))

        return os.path.join(self.SAVE_DIR,f'monte_carlo_b_{self.b:.2f}_k_{self.k:.2f}_harmonic_{int(self.W/self.w0)}.json')

def hyperparameter_search(layers, neurons, learning_rate):
    
    for lr in learning_rate:
            for la in layers:
                for ne in neurons:
                    # s = f'''b_{pinn_params['batch']}_n_{pinn_params['neurons']}_l_{pinn_params['layers']}_mu0_{pinn_params['mu_guess']:.1f}_k0_{pinn_params['k_guess']:.1f}_pys_{int(pinn_params['physics_points'])}_obs_{int(pinn_params['observation_points'])}_iter_{int(pinn_params['epochs']/1000)}k_lr_{pinn_params['learning_rate']:4.2e}_lb_{pinn_params['regularization']:4.2e}'''
                    s = "hyperparameter_search"
                    SAVE_DIR, SAVE_GIF_DIR = DIRs(path_name = s, path_gif_name = None)
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
                        "stop_eps": 5*10**-3,
                        "stop_eps_u": 10**-4,
                        "regularization": 5*10**5,
                        "epochs": 10**5,
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
                    "stop_eps": 5*10**-3,
                    "stop_eps_u": 10**-4,
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

def monte_carlo(N, d = 2, w0 =  20, Harmonic = 3, epochs = 8*10**4):
    
    for count in range(N):
        s = "monte_carlo"
        SAVE_DIR, SAVE_GIF_DIR = DIRs(path_name = s, path_gif_name = None)
        control_params = {  'SEARCH': False,
                            'PLOT': False,
                            'SAVE_DIR': SAVE_DIR,
                            'SAVE_GIF_DIR': SAVE_GIF_DIR,
                            'TEXT': False,
                            'FIGS': -1}

        system_params = {   "t": np.linspace(0,1,5000),
                            "m": 1,
                            "b": 2*d,
                            "k": w0**2,
                            "noise": 0.0,
                            "F0": Harmonic*w0**2,
                            "W": Harmonic*w0,
                            "x0": 1,
                            "x_0": 0}

        pinn_params = {     "physics_points": 500, 
                            "observation_points": 100, 
                            # "physics_points": 2*system_params["W"], # Acording to Nyquist f_sampling >= f_max, but to be better lets take f_sampling = 2*2*pi*f_max = 2*W
                            # "observation_points": 2*system_params["W"]/5, # Let's get 20% of physics points
                            "neurons": 80,
                            "layers": 3,
                            "stop_eps": 5*10**-3,
                            "stop_eps_u": 10**-4,
                            "learning_rate": 10**-3.5,
                            "regularization": 3*10**6,
                            "epochs": epochs,
                            "batch": None,
                            "k_guess": 0.0,
                            "mu_guess": 0.0}

        # Randomly select the initial guesses in the normal distribution centered in the true value
        # with 2 standard deviation of the distance of the total steps times the learning rate ( rough approximation ) 
        pinn_params["mu_guess"] = 2*d + np.random.normal(0, (d/w0)*pinn_params["epochs"]*pinn_params["learning_rate"]/2, 1)[0] # N(K,2*sigma*mu/sqrt(k))
        pinn_params["k_guess"] = w0**2 + np.random.normal(0, pinn_params["epochs"]*pinn_params["learning_rate"]/2, 1)[0] # N(K,2*sigma)

        spring_pinn = PINN(control_params, system_params, pinn_params)
        spring_pinn.train()
        MC_PATH = spring_pinn.save_monte_carlo(pinn_params)
