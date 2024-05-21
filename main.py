from ze import *
from tqdm.auto import tqdm, trange

if __name__ == "__main__":
    set_cuda()
    torch.set_default_dtype(torch.float32)
    
    # layers = [3,6,9]
    # neurons = [40,80,120]
    # learning_rate = [5*10**-2, 5*10**-3, 10**-3, 10**-4, 10**-5, 5*10**-6]
    # hyperparameter_search(layers, neurons, learning_rate)

    # PATH = r"C:\Users\jedua\Documents\INSA\Python\PINN\PINN-SPRING\hyperparameter_search"
    # file = "results.csv"
    # hyper_dict = interpret_hyperparameters(PATH, file)

    N = 50
    H = 3    
    Ds = [2]
    Ws = [20,40,80,160,320] # From 80  ahead data points are insufficient accordint to Nyquist
    # Data points = 100
    # Physics points = 500

    combinations = []
    for i in range(len(Ds)):
        for j in range(len(Ws)):
            combinations.append((Ds[i], Ws[j]))

    for (d,w0) in combinations:
        print(f"\n\t\t-------------------------------d: {d} - w0: {w0}-------------------------------\n")
        monte_carlo(N, d, w0, H, epochs = 8*10**4)

    # PATH = r"C:\Users\jedua\Documents\INSA\Python\PINN\PINN-SPRING\monte_carlo"
    # file = "monte_carlo_b_4.00_k_400.00_harmonic_3.csv"
    # monte_carlo_viz_csv(PATH, file)