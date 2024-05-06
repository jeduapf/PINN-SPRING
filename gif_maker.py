import os
from PIL import Image

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

def read_files_list(PATH):
    files = []
    for x in os.listdir(PATH):
        files.append(os.path.join(PATH,x))

    return files[::5]

if __name__ == "__main__":

    PATH = r"C:\Users\jedua\Documents\INSA\Python\PINN\PINN-SPRING\mu0_-2.0_k0_437.0_pys_900_obs_90_iter_500k_lr_1.00e-04_lb_1.00e+04\gif"
    save_gif_PIL(os.path.join(os.getcwd(),"g.gif"), read_files_list(PATH), fps=5, loop=0)