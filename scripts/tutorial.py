# BO implementation made by following: https://www.youtube.com/watch?v=BQ4kVn-Rt84
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qProbabilityOfImprovement, qUpperConfidenceBound
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.models import ModelListGP, SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls.exact_marginal_log_likelihood import \
    ExactMarginalLogLikelihood


def target_function(points):
    result = []
    for pnt in points:
        x = pnt[0]
        fn = np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1 / (x**2 + 1)
        # f(x) = e^(-(x - 2)^2) + e^((-(x - 6)^2) / 10) + (1 / (x^2+1))
        result.append(fn)
    return torch.tensor(result)

def plot_target_function():
    x = np.linspace(-2.0, 10.0, 100)  # array of 100 evenly spaced points from -2.0 to 10.0
    x_new = x.reshape((100, -1))  # new array shape of 100 rows, 1 column
    y = target_function(x_new)  # type(tensor) enables calculation on GPU
    plt.plot(x, y)
    plt.show()

# to train gaussian process regressors for new iteration point
def generate_random_initial_data(num_points=10):
    train_x = torch.rand(num_points, 1) * 12 - 2  # type(tensor) and num_pointsx1 array of random floats
    exact_obj = target_function(train_x).unsqueeze(-1)  # 1 row of num_points -> num_points columns of 1 (for single-objective optimization)
    best_observed_value = exact_obj.max().item()  # get max value
    return train_x, exact_obj, best_observed_value

def generate_static_initial_data():
    train_x = np.linspace(-2.0, 10.0, 100)  # array of 100 evenly spaced points from -2.0 to 10.0
    train_x = train_x.reshape((100, -1))  # new array shape of 100 rows, 1 column
    train_x = torch.from_numpy(train_x)
    exact_obj = target_function(train_x).unsqueeze(-1)  # type(tensor) enables calculation on GPU
    best_observed_value = exact_obj.max().item()
    return train_x, exact_obj, best_observed_value

def get_next_points(init_x, init_y, best_init_y, bounds, num_points=1):
    single_model = SingleTaskGP(init_x, init_y)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)  # for uncertainty quantification
    fit_gpytorch_model(mll)

    # acq = qExpectedImprovement(model=single_model, best_f=best_init_y)  # for function without noise
    acq = qUpperConfidenceBound(model=single_model, beta=2)
    # acq = qProbabilityOfImprovement(model=single_model, best_f=best_init_y)
    # acq = qMaxValueEntropy(single_model, candidate_set=init_x)


    candidates, _ = optimize_acqf(acq_function=acq,
                                  bounds=bounds, 
                                  q=num_points,
                                  num_restarts=200,
                                  raw_samples=512,
                                  options={"batch_limit": 5, "maxiter": 200})

    return candidates, single_model

plot_target_function()
num_iterations = 10
init_x, init_y, best_init_y = generate_random_initial_data(num_points=10)
bounds = torch.tensor([[0.0], [10.0]])  # one input of domain [0 to 10]
# bounds = touch.tensor([[0.0, 1.0]. [10.0, 9.0]])  # same input as above and another input of domain [1 to 9]

for i in range(num_iterations):
    print(f"Iteration Number: {i + 1}")

    new_candidates, single_model = get_next_points(init_x, init_y, best_init_y, bounds, num_points=1)
    new_results = target_function(new_candidates).unsqueeze(-1)
    print(f"    New candidates are: {new_candidates}")
    print(f"    New results are: {new_results}")

    init_x = torch.cat([init_x, new_candidates])
    init_y = torch.cat([init_y, new_results])

    best_init_y = init_y.max().item()
        
    print(f"    Best point performs this way: {best_init_y}")


test_X = torch.linspace(-2.0, 10.0, 100)
f, ax = plt.subplots(1, 1, figsize=(6, 4))
with torch.no_grad():
    # compute posterior
    posterior = single_model.posterior(test_X)
    # Get upper and lower confidence bounds (2 standard deviations from the mean)
    lower, upper = posterior.mvn.confidence_region()
    # Plot training points as black stars
    ax.plot(init_x.cpu().numpy(), init_y.cpu().numpy(), 'k*')
    # Plot posterior means as blue line
    ax.plot(test_X.cpu().numpy(), posterior.mean.cpu().numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_X.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)
ax.legend(['Observed Data', 'Mean', 'Confidence'])
plt.tight_layout()
plt.show()
