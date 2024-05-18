from tqdm.auto import tqdm, trange
import numpy as np
import torch
from ze.utils import *
from ze.torch_utils import FCN

class PINN():

    def __init__(self, control_params, system_params, pinn_params):

        # Dashboard parameters
        self.PLOT = control_params['PLOT']
        self.SAVE_DIR = control_params['SAVE_DIR']
        self.SAVE_GIF_DIR = control_params['SAVE_GIF_DIR']
        self.TEXT = control_params['TEXT']
        self.FIGS = control_params['FIGS']

        # ---------------------------------------- System ----------------------------------------
        # System parameters
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
        self.observation_points = pinn_params["observation_points"]
        self.neurons = pinn_params["neurons"]
        self.layers = pinn_params["layers"]
        assert isinstance(self.physics_points,int) and isinstance(self.observation_points,int) and (self.physics_points<len(self.t) and self.observation_points<len(self.t)), "Amount of data and physics points must be an integer smaller than the total amout of points in the simulation !"
        
        # PINN neural network
        self.pinn = FCN(1,1,self.neurons,self.layers)

        # PINN optimization parameters
        self.learning_rate = pinn_params["learning_rate"]
        self.regularization = pinn_params["regularization"]
        self.epochs = pinn_params["epochs"]
        self.batch = pinn_params["batch"]
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

    def physics_loss(self, t_physics, force):
        # PHYSICS LOSS
        u_phy_hat = self.pinn(t_physics)
        dudt = torch.autograd.grad( u_phy_hat, self.t_physics, 
                                    torch.ones_like(u_phy_hat), 
                                    create_graph=True)[0]

        d2udt2 = torch.autograd.grad(   dudt, self.t_physics, 
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

    def step(self, i):
        self.optimiser.zero_grad()

        phy_loss = self.physics_loss(self.t_physics, self.force)
        dat_loss = self.data_loss(self.t_obs, self.u_obs)
        loss = phy_loss + self.regularization*dat_loss

        loss.backward()
        self.optimiser.step()

        self.losses.append([phy_loss.item(),
                            dat_loss.item(),
                            loss.item()])

        # plot the result as training progresses
        if i % self.FIGS == 0:

            if self.TEXT:
                tqdm.write(f"{i}\n>>Physics: {self.losses[-1][0]:.3f} >>Data: {self.losses[-1][1]:.3f} >>Total: {self.losses[-1][2]:.3f}\n>>Mu: {self.constants[-1][0]:.3f} >>k: {self.constants[-1][1]:.3f}\n")

            if self.PLOT:
                fig = plt.figure(figsize=(12,5))

                u = self.pinn(self.t_test)
                plt.scatter(self.t_obs_np, self.u_obs_np, label="Noisy observations", alpha=0.6)
                plt.plot(self.t_test_np, u.detach().cpu().numpy(), label="PINN solution", color="tab:green")
                plt.title(f"Training step {i}")
                plt.legend()

                file = os.path.join(self.SAVE_GIF_DIR,"pinn_%.8i.png"%(i+1))
                plt.savefig(file, dpi=100, facecolor="white")
                self.files.append(file)
                plt.close(fig)

    def train(self, eps = 6*10**-4):

        if self.FIGS is None:
            self.FIGS = int(self.epochs/100)

        if self.batch is None:
            bar = trange(self.epochs)
            for i in bar:
                self.step(i)

                if self.losses[-1][0] + self.losses[-1][1] < eps*self.losses[0][2] and torch.abs(self.constants[-1][0] - self.b_torch) < eps and torch.abs(self.constants[-1][1] - self.k_torch) < eps:
                    print("\n\n\t\t Converged, finishing early !\n\n")
                    break
        else:
            assert isinstance(self.batch,int) and self.batch > 0 , "Batch size must be a positive integer..."
            pass



    def save_plots(self):
        files1, files2 = write_losses(  self.u_obs, self.derivatives, self.constants, 
                                        self.SAVE_DIR, self.losses, self.force_np, 
                                        l = self.regularization, TEXT = False, PLOT = True, 
                                        fig_pass = self.FIGS, SAVE_PATH = self.SAVE_DIR)

        losses_constants_plot(self.constants, self.losses, self.SAVE_DIR, self.d, self.w0)

        print("\n\nGenerating GIFs...\n\n")
        save_gif_PIL(os.path.join(self.SAVE_DIR,"learning_k_mu.gif"), self.files, fps=60, loop=0)
        save_gif_PIL(os.path.join(self.SAVE_DIR,"loss1.gif"), files1, fps=60, loop=0)
        save_gif_PIL(os.path.join(self.SAVE_DIR,"loss2.gif"), files2, fps=60, loop=0)

    def predict(self, t):

        return self.pinn(t)
