import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
torch.set_default_device(device)
print(f"\nUsing device: {device}\n\n")

from forced_spring_sol import forced_damped_spring

def plot_initial(t_obs_np, u_obs_np, t_physics_np, sys_params, SAVE_DIR):

    u_physics_np = forced_damped_spring(t_physics_np,
                                        sys_params['m'],
                                        sys_params['b'],
                                        sys_params['k'],
                                        sys_params['f0'],
                                        sys_params['ffreq'],
                                        sys_params['x0'],
                                        sys_params['x_0'])

    t = np.linspace(0,1,10**5)
    applied_force = sys_params['f0']*np.cos(sys_params['ffreq']*t)
    u = forced_damped_spring(t,
                            sys_params['m'],
                            sys_params['b'],
                            sys_params['k'],
                            sys_params['f0'],
                            sys_params['ffreq'],
                            sys_params['x0'],
                            sys_params['x_0'])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=( "Initial points", "Input force and system's output"))

    fig.add_trace(
        go.Scatter(x=t_obs_np, y=u_obs_np, mode='markers', name = f"PINN observation points (measured data - {len(t_obs_np)} points) "),
        row=1, col=1
        )
    fig.add_trace(
        go.Scatter(x=t_physics_np, y=np.zeros_like(t_physics_np), mode='markers', name = f"PINN physics points (not measured data - {len(t_physics_np)} points)"),
        row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=t, y=applied_force, name = "Input force"),
        row=2, col=1
        )
    fig.add_trace(
        go.Scatter(x=t, y=sys_params['f0']*u, name = f"System output position (multiplied by {sys_params['f0']})"),
        row=2, col=1
        )

    fig.write_html(os.path.join(SAVE_DIR,"Initial.html"))

def losses_constants_plot(mus, losses, SAVE_DIR, d = 2, w0 = 20):
    plt.rcParams['figure.figsize'] = [18, 9]

    mus_np = np.array(mus)
    losses_np = np.array(losses)


    X = np.linspace(0,mus_np.shape[0],mus_np.shape[0])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=( "Mu", "K"))

    fig.add_trace(
        go.Scatter(x=X, y=mus_np[:,0], name = "PINN estimate"),
        row=1, col=1
        )
    fig.add_hline(y=2*d, line_dash="dot", name = f"Expected mu {2*d:.2f}", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=X, y=mus_np[:,1], name = "K"),
        row=2, col=1
        )
    fig.add_hline(y=w0**2, line_dash="dot", name = f"Expected k {w0**2:.2f}", row=2, col=1)

    fig.write_html(os.path.join(SAVE_DIR,"constants.html"))


    # fig = plt.figure()
    # plt.title("mu")
    # plt.plot(list(mus_np[:,0]), label="PINN estimate")
    # plt.hlines(2*d, 0, len(mus_np), label="True value", color="tab:green")
    # plt.legend()
    # plt.xlabel("Training step")
    # plt.savefig(os.path.join(SAVE_DIR,"mu.png"), dpi=100, facecolor="white")
    # plt.close(fig)

    # fig = plt.figure()
    # plt.title("k")
    # plt.plot(list(mus_np[:,1]), label="PINN estimate")
    # plt.hlines(w0**2, 0, len(mus_np), label="True value", color="tab:green")
    # plt.legend()
    # plt.xlabel("Training step")
    # plt.savefig(os.path.join(SAVE_DIR,"k.png"), dpi=100, facecolor="white")
    # plt.close(fig)

    # fig, ax = plt.subplots(3,1)
    # fig.suptitle(f"Losses")
    # plt.subplot(3,1,1)
    # plt.plot(list(losses_np[:,0]), label="Physical loss")
    # plt.legend()
    # plt.subplot(3,1,2)
    # plt.plot(list(losses_np[:,1]), label="Data loss")
    # plt.legend()
    # plt.subplot(3,1,3)
    # plt.plot(list(losses_np[:,2]), label="Total loss")
    # plt.legend()
    # plt.xlabel("Training step")
    # plt.tight_layout()
    # plt.savefig(os.path.join(SAVE_DIR,"Losses.png"), dpi=100, facecolor="white")
    # plt.close(fig)
        
    X = np.linspace(0,losses_np.shape[0],losses_np.shape[0])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        subplot_titles=( "Loss f", "Loss Data", "Total Loss"))

    fig.add_trace(
        go.Scatter(x=X, y=losses_np[:,0], name = "Physical Loss"),
        row=1, col=1
        )
    fig.add_trace(
        go.Scatter(x=X, y=losses_np[:,1], name = "Data Loss"),
        row=2, col=1
        )

    fig.add_trace(
        go.Scatter(x=X, y=losses_np[:,2], name = "Total Loss"),
        row=3, col=1
        )
    fig.update_layout( height=1500, width=1500, 
                            title_text="Losses training", showlegend=False)

    fig.write_html(os.path.join(SAVE_DIR,"losses.html"))

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

def exact_solution(d, w0, t):
    "Defines the analytical solution to the under-damped harmonic oscillator problem above."

    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*t)
    exp = torch.exp(-d*t)
    u = exp*2*A*cos

    return u

def DIRs(path_name = 'save_imgs', path_gif_name = 'gif'):
    
    current_dir = os.getcwd()
    SAVE_DIR = os.path.join(current_dir,path_name)
    try:
        os.mkdir(SAVE_DIR)
    except:
        pass

    SAVE_GIF_DIR = os.path.join(SAVE_DIR,path_gif_name)
    try:
        os.mkdir(SAVE_GIF_DIR)
    except:
        pass

    return SAVE_DIR, SAVE_GIF_DIR

class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

def write_losses(u_obs, us, mus, FINAL_DIR, losses, F, l = 10**4, TEXT = False, PLOT = True, fig_pass = 13, SAVE_PATH = r'.'):
    SAVE_DIR = os.path.join(SAVE_PATH,'figures')
    try:
        os.mkdir(SAVE_DIR)
    except:
        pass

    plt.rcParams['figure.figsize'] = [18, 9]
    if TEXT:
        f = open(os.path.join(SAVE_DIR, "results.txt"), "w")


    u_obs = u_obs.detach().cpu().numpy().squeeze()

    print(f"\n\n\t\t\t Saving images...")
    files1 = []
    files2 = []
    L1s = []
    L2s = []
    pytorch_l1 = []
    pytorch_l2 = []
    pytorch_l = []
    for i in tqdm(range(len(us))):
        item = (us[i],mus[i],losses[i])


        d2udt2 = item[0][0]
        dudt = item[0][1]
        u_phy_hat = item[0][2]
        u_obs_hat = item[0][3]

        mu = item[1][0]
        k = item[1][1]

        # loss1 = torch.mean((d2udt2 + mu_nn*dudt + k_nn*u)**2)
        loss1 = item[2][0]
        # loss2 = torch.mean((u - u_obs)**2)
        loss2 = item[2][1]
        # loss = loss1 + lambda1*loss2
        loss = item[2][2]

        pytorch_l1.append(loss1)
        pytorch_l2.append(loss2)
        pytorch_l.append(loss)

        # Recalculated losses
        L1 = np.mean((d2udt2 + mu*dudt + k*u_phy_hat)**2)
        L2 = np.mean((u_obs_hat - u_obs)**2)
        L = L1 + l*L2
        L1s.append(L1)
        L2s.append(L2)
        
        if PLOT: 
            if i % fig_pass ==0:
                fig, ax = plt.subplots(5,1)

                fig.suptitle(f"Physics loss iter: {i}")
                plt.subplot(5,1,1)
                plt.plot(d2udt2, label="u2_dt2")
                plt.legend()

                plt.subplot(5,1,2)
                plt.plot(mu*dudt, label=f"mu*u_dt  mu = {mu:.5f}")
                plt.legend()

                plt.subplot(5,1,3)
                plt.plot(k*u_phy_hat, label=f"k*u  k = {k:.5f}")
                plt.legend()

                plt.subplot(5,1,4)
                plt.plot( d2udt2 + mu*dudt + k*u_phy_hat, label="d2udt2 + mu*dudt + k*u_phy_hat")
                plt.plot( -F, label="-F")
                plt.legend()

                plt.subplot(5,1,5)
                plt.plot((d2udt2 + mu*dudt + k*u_phy_hat - F)**2, label="|u2_dt2 + mu*u_dt + k*u_pinn - F|^2")
                plt.legend()

                plt.xlabel(f"Training points ({len(d2udt2)})")
                plt.tight_layout()
                file = os.path.join(SAVE_DIR,f"loss1_iter_{i}.png")
                files1.append(file)
                plt.savefig(file, dpi=100, facecolor="white")
                plt.close(fig)

                fig = plt.figure()
                plt.title(f"Data loss iter: {i}")
                plt.plot(u_obs, label="u_obs", linewidth=2, color='blue', alpha=0.3)
                plt.plot(u_obs_hat, label="u_obs_hat", linewidth=2, color='red', alpha=0.3)
                plt.plot((u_obs_hat - u_obs)**2, label="|u_obs_hat - mu*u_obs|^2", linewidth=2, color='green', alpha=0.95)
                plt.xlabel(f"Data points {len(u_obs_hat)}")
                plt.legend()
                plt.tight_layout()

                file = os.path.join(SAVE_DIR,f"loss2_iter_{i}.png")
                files2.append(file)
                plt.savefig(file, dpi=100, facecolor="white")
                plt.close(fig)


        if TEXT :
            f.write(f"\n-*-*-*-*-*-*-*-*- Iteration: {i} -*-*-*-*-*-*-*-*-\n")
            f.write(f"\n***************************** LOSS 1 *****************************\n")
            for count in range(len(d2udt2)):
                f.write(f"{d2udt2[count]:.5f}, {mu:.5f}, {dudt[count]:.5f}, {k:.5f}, {u_phy_hat[count]:.5f}\n")

            f.write(f"\n***************************** Result *****************************\n")
            f.write(f"{L1}")

            f.write(f"\n***************************** LOSS 2 *****************************\n")
            for count in range(len(u_obs_hat)):
                f.write(f"{u_obs_hat[count]:.5f}, {u_obs[count]:.5f}\n")

            f.write(f"\n***************************** Result *****************************\n")
            f.write(f"{L2}")

            f.write(f"\n***************************** FINAL LOSS *****************************\n")
            f.write(f"{L}")
            f.write("\n\n")
    
    if TEXT:
        f.close()

    # if PLOT:
    #     L1s = np.array(L1s)
    #     L2s = np.array(L2s)
    #     pytorch_l1 = np.array(pytorch_l1)
    #     pytorch_l2 = np.array(pytorch_l2)
    #     pytorch_l = np.array(pytorch_l)

    #     fig, ax = plt.subplots(3,1)
    #     fig.suptitle(f"Difference between losses calculated pytorch and numpy")

    #     plt.subplot(3,1,1)
    #     # plt.plot(pytorch_l1, label="Pytorch Loss 1", linewidth=2, color='red', alpha=0.3)
    #     # plt.plot(L1s, label="Numpy Loss 1", linewidth=2, color='blue', alpha=0.8)
    #     plt.plot(np.abs(pytorch_l1 - L1s), linewidth=2 , label=f"Physics Loss Sum: {np.sum(np.abs(pytorch_l1 - L1s)):.5f}")
    #     plt.legend()

    #     plt.subplot(3,1,2)
    #     # plt.plot(pytorch_l2, label="Pytorch Loss 2", linewidth=2, color='red', alpha=0.3)
    #     # plt.plot(L2s, label="Numpy Loss 2", linewidth=2, color='blue', alpha=0.8)
    #     plt.plot(np.abs(pytorch_l2 - L2s), linewidth=2 , label=f"Data Loss Sum: {np.sum(np.abs(pytorch_l2 - L2s)):.5f}")
    #     plt.legend()

    #     plt.subplot(3,1,3)
    #     # plt.plot(pytorch_l, label="Pytorch Loss", linewidth=2, color='red', alpha=0.3)
    #     # plt.plot(L1s + l*L2s, label="Numpy Loss", linewidth=2, color='blue', alpha=0.8)
    #     plt.plot(np.linspace(0,len(pytorch_l),len(pytorch_l)),np.abs(pytorch_l - (L1s + l*L2s)), linewidth=2 , label=f"All Loss Sum: {np.sum(np.abs(pytorch_l - (L1s + l*L2s))):.5f}")

    #     plt.xlabel(f"Iterations")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(FINAL_DIR,f"ALL.png"), dpi=300, facecolor="white")
    #     plt.close(fig)

    return files1, files2

if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = [12, 5]
    torch.manual_seed(123)

    # System dynamics
    d, w0 = 2, 20
    noise = 0.0

    # Imposing force
    F0 = 250 # Amplitude (in the signal will be divided by w0**2)
    W = 5*w0  # Angular frequency

    w_max = np.max([w0, W])

    # System initial conditions
    x0 = 1
    x_0 = 0

    sys_params = {
                    'f0': F0,
                    'ffreq' : W,
                    'x0': x0,
                    'x_0': x_0,
                    'm': 1,
                    'b' : 2*d,
                    'k':w0**2,
    }

    # Shuold be at least 2 for Nyquist, but can be higher for safety results
    K_freq = 2*15

    N_obs_points = K_freq*int(w_max/6)          # Related to maximum frequency of the signal
    N_phy_points = K_freq*N_obs_points          # Related to physics loss and continuity ( smoothness ) of the PINN solution
    
    N_obs_points = 50
    N_phy_points = 300

    neurons = 80                                # Related to maximum frequency too?  NO apperantly 
    layers = 3

    t_obs = torch.rand(N_obs_points).view(-1,1)
    t_obs_np = t_obs.detach().cpu().numpy()

    u_obs_np = forced_damped_spring(t_obs_np, m = 1, b = 2*d, k = w0**2, f0 = F0, ffreq = W, x0 = x0, x_0 = x_0, eps = 10**-8) + noise*np.random.randn(t_obs_np.shape[0],t_obs_np.shape[1])
    u_obs = torch.tensor(u_obs_np)
    
    t_test = torch.linspace(0,1,1000).view(-1,1)
    t_test_np = t_test.detach().cpu().numpy()

    u_exact = torch.tensor(forced_damped_spring(t_test_np, m = 1, b = 2*d, k = w0**2, f0 = F0, ffreq = W, x0 = x0, x_0 = x_0, eps = 10**-8))

    print(f"True value of mu: {2*d}")
    print(f"True value of k: {w0**2}\n\n")

    # TODO: correct this idea
    lr = 10**(-np.log(w_max))        
    lambda1 = int(10**( np.log(w_max) + (K_freq+layers)/K_freq ))       # Related to maximum frequency of the signal, cuz thz higher the frequency the higher the derivative !

    lr = 7*10**-4
    lambda1 = 5*10**5

    # Related to the learning rate and initial guess error steps and some confidence (99,7% 3*sig ?)
    EPOCHS = int(1/lr) 
    # EPOCHS = 5*10**4

    EPOCHS = 200
    figs = int(EPOCHS/100)

    # Error related to constants of the system 'guess'
    # to have 95% confidence the model can solve in the amount of steps provided and supposing
    # it goes in the bad direction for half the way 2*sigma ( think how to solve it )
    MU, SIG = 0, EPOCHS*lr/2
    start_mu = 2*d + np.random.normal(MU, SIG, 1)[0]
    start_k = w0**2 + np.random.normal(MU, SIG, 1)[0]

    # start_k = 413
    # start_mu = -5


    SAVE_DIR, SAVE_GIF_DIR = DIRs(path_name = f'mu0_{start_mu:.1f}_k0_{start_k:.1f}_pys_{int(N_phy_points)}_obs_{int(N_obs_points)}_iter_{int(EPOCHS/1000)}k_lr_{lr:4.2e}_lb_{lambda1:4.2e}', 
                                    path_gif_name = 'gif')

    # define a neural network to train
    # N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS
    pinn = FCN(1,1,neurons,layers)

    # define training points over the entire domain, for the physics loss
    t_physics = torch.linspace(0,1,N_phy_points).view(-1,1).requires_grad_(True)
    t_physics_np = t_physics.detach().cpu().numpy()
 
    Force = torch.tensor( sys_params['f0'] * np.cos(sys_params['ffreq']*t_physics_np)/sys_params['m'] )
    Force_np = Force.detach().cpu().numpy().squeeze()

    # train the PINN
    mu, k = 2*d, w0**2

    # treat mu as a learnable parameter
    mu_nn = torch.nn.Parameter(torch.tensor([float(start_mu)], requires_grad=True))
    k_nn = torch.nn.Parameter(torch.tensor([float(start_k)], requires_grad=True))

    mus = []
    us = []
    files = []
    losses=[]

    plot_initial(   t_obs_np.squeeze(), 
                    u_obs_np.squeeze(), 
                    t_physics_np.squeeze(), 
                    sys_params,
                    SAVE_DIR)
    

    S = t_physics.size()[0]
    batch_size = int((10/100)*S)
    optimiser = torch.optim.Adam(list(pinn.parameters())+[mu_nn, k_nn],lr=lr, betas=(0.95, 0.999))

    for i in tqdm(range(EPOCHS)):
        

        # batch the physics points
        permutation = torch.randperm(S)





        for _i in range(0, S, batch_size):

            optimiser.zero_grad()

            # compute each term of the PINN loss function above
            # using the following hyperparameters:
            # compute physics loss
            u_phy_hat = pinn(t_physics)
            dudt = torch.autograd.grad(u_phy_hat, t_physics, torch.ones_like(u_phy_hat), create_graph=True)[0]
            d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
            # loss1 = torch.mean((d2udt2 + mu_nn*dudt + k_nn*u_phy_hat)**2)

            # Now with the imposed force term
            loss1 = torch.mean((d2udt2 + mu_nn*dudt + k_nn*u_phy_hat - Force)**2)

            # compute data loss
            u_obs_hat = pinn(t_obs)
            loss2 = torch.mean((u_obs_hat - u_obs)**2)

            # backpropagate joint loss, take optimiser step
            loss = loss1 + lambda1*loss2 
            
            loss.backward()
            optimiser.step()

            # record mu value
            mus.append([mu_nn.item(),
                        k_nn.item()
                        ])

            us.append([d2udt2.detach().cpu().numpy().squeeze(),
                        dudt.detach().cpu().numpy().squeeze(),
                        u_phy_hat.detach().cpu().numpy().squeeze(),
                        u_obs_hat.detach().cpu().numpy().squeeze(),
                        ])

            losses.append([ loss1.item(),
                            loss2.item(),
                            loss.item()])

        # plot the result as training progresses
        if i % figs == 0:
            fig = plt.figure(figsize=(12,5))
            # fig.set_size_inches(18.5, 10.5)
            u = pinn(t_test).detach()
            plt.scatter(t_obs_np[:,0], u_obs_np[:,0], label="Noisy observations", alpha=0.6)
            plt.plot(t_test_np[:,0], u.detach().cpu().numpy()[:,0], label="PINN solution", color="tab:green")
            plt.title(f"Training step {i}")
            plt.legend()

            file = os.path.join(SAVE_GIF_DIR,"pinn_%.8i.png"%(i+1))
            plt.savefig(file, dpi=100, facecolor="white")
            files.append(file)
            plt.close(fig)

    files1, files2 = write_losses(u_obs, us, mus, SAVE_DIR, losses, Force_np, l = lambda1, TEXT = False, PLOT = True, fig_pass = figs, SAVE_PATH = SAVE_DIR)
    losses_constants_plot(mus, losses, SAVE_DIR, d, w0)

    print("\n\nGenerating GIFs...\n\n")
    save_gif_PIL(os.path.join(SAVE_DIR,"learning_k_mu.gif"), files, fps=60, loop=0)
    save_gif_PIL(os.path.join(SAVE_DIR,"loss1.gif"), files1, fps=60, loop=0)
    save_gif_PIL(os.path.join(SAVE_DIR,"loss2.gif"), files2, fps=60, loop=0)