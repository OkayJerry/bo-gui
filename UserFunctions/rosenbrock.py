import numpy as np

def rosenbrock(x_decision, x_env=None, noise=.01):
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