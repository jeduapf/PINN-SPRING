from ze import *

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

    N = 20
    Ds = [2,5]
    Ws = [20,40,80]

    combinations = []
    for i in range(len(Ds)):
        for j in range(len(Ws)):
            combinations.append((Ds[i], Ws[j]))

    for (d,w0) in combinations:
        print(f"\nd: {d} \t w0: {w0}\n")
        monte_carlo(2, d = 2, w0 =  20, Harmonic = 3)