import numpy as np


def rosenbrock(x_decision, x_env=None, noise=.01):
    """ This is a classic non-convex function that is often used for testing 
    optimization algorithms. 

    The function is defined as: f(x,y) = (a - x)^2 + b(y - x^2)^2
    """
    f = 0
    ndim = len(x_decision)
    for i in range(len(x_decision) -1):
        f += (x_decision[i+1]-x_decision[i]**2)**2 + 0.01*(1-x_decision[i])**2
    if x_env is not None:
        if type(x_env)==float:
            f += (x_env-x_decision[-1]**2)**2 + 0.01*(1-x_decision[-1])**2
            nenv = 1
        else:
            f += (x_env[0]-x_decision[-1]**2)**2 + 0.01*(1-x_decision[-1])**2
            for i in range(1,len(x_env) -1):
                f += (x_env[i+1]-x_env[i]**2)**2 + 0.01*(1-x_env[i])**2
            nenv = len(x_env)
    else:
        nenv = 0

    return f/(ndim+nenv)  + np.random.randn()*noise

def levy(x):
    """This is a high-dimensional function that is commonly used to test 
    optimization algorithms. 
    
    Where `x` is the input vector.The function is defined as:
    f(x) = sin^2(pi*x(1) + 1) + sum[(xi - 1)^2 * (1 + 10sin^2(pi*xi + 1))]
    """
    w = 1 + (x - 1) / 4.0  # scaling
    term1 = np.sin(np.pi * w[0])**2
    term2 = (w[1:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[1:-1] + 1)**2)
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + np.sum(term2) + term3

def prior_levy(X):
    return np.array([levy(x) for x in X])

def rastrigin(X, noise=0.01):
    print(X)
    # print(len(X))
    # print(X.shape)
    # print(X.ndim)
    DIM = len(X)
    return (DIM + np.sum(30*X**2 - 10*np.cos(2 * np.pi * X))) + np.random.randn()*noise