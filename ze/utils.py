import scipy.io
import numpy as np
from PIL import Image
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import csv
import re

def forced_damped_spring(t, system_params, eps = 10**-9):
    # System parameters
    m = system_params["m"]
    b = system_params["b"]
    k = system_params["k"]

    # Forced input parameters
    f0 = system_params["F0"]
    ffreq = system_params["W"]
    x0 = system_params["x0"]
    x_0 = system_params["x_0"]

    ksi = b/2/np.sqrt(m*k)
    w0 = np.sqrt(k/m)

    # --------- Homogeneous solution ---------
    
    # Amortecimento critico ( equal real exponential solutions )
    if np.abs(ksi - 1) < eps :
        xh = np.exp(-w0*t)*( x0*np.ones(t.shape) + (x_0 + w0*x0)*t )
        f1 = w0
        f2 = w0 

    # Superamortecido ( real exponential solutions )
    elif ksi - 1 > eps :
        wd = w0*np.sqrt(ksi**2-1)
        xh = np.exp(-ksi*w0*t)* ( ((x_0 + ksi*x0*w0)/wd)*np.sinh(wd*t) + x0*np.cosh(wd*t)  )
        f1 = -ksi*w0 + wd
        f2 = -ksi*w0 - wd
    
    # Subamortecido ( compelx exponential solutions )
    elif ksi - 1 < eps :
        wd = w0*np.sqrt(1-ksi**2)
        xh = np.exp(-ksi*w0*t)* ( ((x_0 + ksi*x0*w0)/wd)*np.sin(wd*t) + x0*np.cos(wd*t)  )
        f1 = -ksi*w0 + wd
        f2 = -ksi*w0 - wd

    else:
        xh = None

    # --------- Forced sinousoidal solution ---------
    
    xf = ( f0/( (2*ksi*w0*ffreq)**2 + (w0**2 - ffreq**2)**2 ) )*( 2*ksi*w0*ffreq*np.sin(ffreq*t) + (w0**2 - ffreq**2)*np.cos(ffreq*t) )
    
    x = xh + xf

    return x.astype(float)

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

    if path_gif_name is not None:
        SAVE_GIF_DIR = os.path.join(SAVE_DIR,path_gif_name)
        try:
            os.mkdir(SAVE_GIF_DIR)
        except:
            pass
    else:
        SAVE_GIF_DIR = None

    return SAVE_DIR, SAVE_GIF_DIR

def read_files_list(PATH):
    files = []
    for x in os.listdir(PATH):
        files.append(os.path.join(PATH,x))

    return files[::5]

def gif_maker(PATH):
    save_gif_PIL(os.path.join(os.getcwd(),"g.gif"), read_files_list(PATH), fps=5, loop=0)

def plot_initial(t_obs_np, u_obs_np, t_physics_np, sys_params, SAVE_DIR):

    u_physics_np = forced_damped_spring(t_physics_np,sys_params)

    t = np.linspace(0,1,10**5)
    applied_force = sys_params['F0']*np.cos(sys_params['W']*t)
    u = forced_damped_spring(t, sys_params)

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
        go.Scatter(x=t, y=sys_params['F0']*u, name = f"System output position (multiplied by {sys_params['F0']})"),
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

def interpret_hyperparameters(PATH, file):
    df = pd.read_csv(os.path.join(PATH,file), header = None)

    x = df.iloc[:,3] # Neurons
    y = df.iloc[:,2] # Layers
    z = np.log10(df.iloc[:,0]) # Learning rate

    fig = make_subplots(rows=1, cols=3,
                    specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}]],
                    subplot_titles=['Relative Error K', 'Relative Error Mu', 'Error u'],
                    )

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',marker=dict(size=12, color=df.iloc[:,4], colorscale='Jet', opacity=0.8, colorbar=dict(thickness=20,len=0.5, x=0.305))), 1, 1)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',marker=dict(size=12, color=df.iloc[:,5], colorscale='Jet', opacity=0.8, colorbar=dict(thickness=20,len=0.5, x=0.655))), 1, 2)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',marker=dict(size=12, color=df.iloc[:,6], colorscale='Jet', opacity=0.8, colorbar=dict(thickness=20,len=0.5))), 1, 3)

    a = 2.2
    fig.update_layout(title='Error relation to hyperparameter', autosize=False,
                      width=1400, height=700,
                      showlegend=False,
                      margin=dict(l=65, r=50, b=65, t=90),
                      scene=dict(
                                camera_eye=dict(x=1.25*a, y=1.25*a, z=1),
                                xaxis_title_text='Neurons',
                                yaxis_title_text='Layers',
                                zaxis_title_text='log(Learning Rate)'),
                      scene2=dict(
                                camera_eye=dict(x=a, y=a, z=1),
                                xaxis_title_text='Neurons',
                                yaxis_title_text='Layers',
                                zaxis_title_text='log(Learning Rate)'),
                      scene3=dict(
                                camera_eye=dict(x=a, y=a, z=1),
                                xaxis_title_text='Neurons',
                                yaxis_title_text='Layers',
                                zaxis_title_text='log(Learning Rate)'))

    fig.write_html(f"{os.path.join(PATH,'errors')}.html")

    return df

def ellipse(x_center=0, y_center=0, ax1 = [1, 0],  ax2 = [0,1], a=1, b =1,  N=100):
   # x_center, y_center the coordinates of ellipse center
   # ax1 ax2 two orthonormal vectors representing the ellipse axis directions
   # a, b the ellipse parameters
   if np.linalg.norm(ax1) != 1 or np.linalg.norm(ax2) != 1:
       raise ValueError('ax1, ax2 must be unit vectors')
   if  abs(np.dot(ax1, ax2)) > 1e-06:
       raise ValueError('ax1, ax2 must be orthogonal vectors')
   #rotation matrix   
   R = np.array([ax1, ax2]).T
   if np.linalg.det(R) <0: 
       raise ValueError("the det(R) must be positive to get a  positively oriented ellipse reference frame")
   t = np.linspace(0, 2*np.pi, N)
   #ellipse parameterization with respect to a system of axes of directions a1, a2
   xs = a * np.cos(t)
   ys = b * np.sin(t)
   
   # coordinate of the  ellipse points with respect to the system of axes [1, 0], [0,1] with origin (0,0)
   xp, yp = np.dot(R, [xs, ys])
   x = xp + x_center 
   y = yp + y_center
   return x, y

def monte_carlo_viz(PATH, f):

    # Val_error, true_b, true_k, pinn_b, pinn_k
    df = pd.read_csv(os.path.join(PATH,f), header = None)
    better_list = []
    for i in range(df.shape[0]) :
        num = re.findall(r'[-0-9]+\.?[0-9]+|[0-9]+\.?[0-9]+', df.iloc[i,3])
        init_b = float(num[0])
        last_b = float(num[-1])

        num = re.findall(r'[-0-9]+\.?[0-9]+|[0-9]+\.?[0-9]+', df.iloc[i,4])
        init_k = float(num[0])
        last_k = float(num[-1])

        better_list.append([df.iloc[i,0], df.iloc[i,1], df.iloc[i,2], init_b, init_k, last_b, last_k])

    arr = np.array(better_list)
    inits_b = arr[:,3]
    inits_k = arr[:,4]
    lasts_b = arr[:,5]
    lasts_k = arr[:,6]

    x = inits_b # b
    y = inits_k # k
    z = -1*np.ones(inits_b.shape) # Learning rate

    x_ = lasts_b # b
    y_ = lasts_k # k
    z_ = np.ones(inits_b.shape) # Learning rate

    sigma_2 = 10**-3.5*8*10**4 
    p = better_list[0][1]/np.sqrt(better_list[0][2])

    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers', name="Initial iteration")])
    fig.add_trace(go.Scatter(x=x_, y=y_, mode='markers', name="Final iteration"))
    x_e, y_e = ellipse(x_center=better_list[0][1], y_center=better_list[0][2], a=sigma_2, b = sigma_2*p)
    x_e_good, y_e_good = ellipse(x_center=better_list[0][1], y_center=better_list[0][2], a=better_list[0][1]*0.01, b = better_list[0][2]*0.01)
    fig.add_trace(go.Scatter(x=x_e, y=y_e, mode='lines', name="Initial", opacity=0.9))
    fig.add_trace(go.Scatter(x=x_e_good, y=y_e_good, mode='lines', name="Good", opacity=0.9))

    fig.update_layout(  title=f'Monte carlo of {len(better_list)} points', autosize=False,
                        xaxis_title="b",
                        yaxis_title="k",
                        width=1400, height=700,
                        showlegend=True)


    fig.write_html(os.path.join(PATH, f[:-4]+'.html'))

    return better_list

