from PyQt5.QtCore import pyqtSignal, QObject
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as copy
import os
from datetime import date
import pickle
import time
import sys

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.optim import optimize_acqf


from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union
from torch import Tensor

import warnings
warnings.filterwarnings("ignore")

def addColorBar(ax, data, y_grid):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import globals as glb

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    num_ticks = 5
    max_ygrid = max(y_grid)
    min_ygrid = min(y_grid)
    range_ygrid = max_ygrid - min_ygrid
    tick_difference = range_ygrid / (num_ticks - 1)
    ticks = [max_ygrid]
    for i in range(num_ticks):
        ticks.append(max_ygrid - i * tick_difference)
    cbar = ax.get_figure().colorbar(data, cax=cax, ticks=ticks)
    
    if ax in glb.main_window.canvas_colorbars.keys():
        glb.main_window.canvas_colorbars[ax].remove()
                
    glb.main_window.canvas_colorbars[ax] = cbar

class simpleNN(torch.nn.Module):
    def __init__(self, n_decision, n_env,
                 list_of_nodes,
                 bounds=None,
                 activation=torch.nn.ELU(),
                 ):
        super(simpleNN, self).__init__()

        self.n_decision = n_decision
        self.n_known_env = n_env
        if bounds is not None:
            self.bounds = torch.tensor(bounds, dtype=torch.float64)
        else:
            self.bounds = None

        self.seq = [torch.nn.Linear(
            n_decision + n_env, list_of_nodes[0]), activation]
        for i in range(1, len(list_of_nodes) - 1):
            self.seq += [torch.nn.Linear(list_of_nodes[i - 1],
                                         list_of_nodes[i]), activation]
        self.seq += [torch.nn.Linear(list_of_nodes[-2], list_of_nodes[-1])]
        self.seq = torch.nn.Sequential(*self.seq)

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def forward(self, x_decision, x_env=None):
        if self.n_known_env > 0:
            x = torch.cat((x_decision, x_env), 1)
        else:
            x = x_decision
        if self.bounds is not None:
            x = self.normalize(x)

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])
        return self.seq(x)

    def normalize(self, x):
        with torch.no_grad():
            diff = self.bounds[:, 1] - self.bounds[:, 0]
            out = 2 * (x.to(torch.float64) - self.bounds[:, 0]) / diff - 1.0

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])
        return out.to(torch.float32)


mse = torch.nn.MSELoss()


def train_simpleNN(model, x_decision, x_env, y,
                   lr=None, lr_func=None, optimizer=None, optim_kwargs=None,
                   epochs=None, maxtime=None,
                   device=None, hist=None,
                   ):

    assert lr is not None or lr_func is not None
    lr = lr or lr_func(0)

    assert epochs is not None or maxtime is not None
    epochs = epochs or 999999
    maxtime = maxtime or 999999

    hist = hist or {'train_loss': []}

    t0 = time.time()
    model.train()

    x_decision = torch.tensor(x_decision, dtype=torch.float32).to(device)
    if x_env is not None:
        x_env = torch.tensor(x_env, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    optim_kwargs = optim_kwargs or {}
    opt = optimizer or torch.optim.Adam(model.parameters(filter(lambda p: p.requires_grad, model.parameters())),
                                        lr=lr, **optim_kwargs)

    min_loss = 9999.9
    checkpoint = None

    for epoch in range(epochs):
        opt.zero_grad()
        y_pred = model(x_decision, x_env)
        loss = mse(y, y_pred)
        loss.backward()
        opt.step()
        loss = loss.item()
        if loss < min_loss:
            min_loss = loss
            checkpoint = model.state_dict().copy()
        hist['train_loss'].append(loss)
        if time.time() - t0 > maxtime:
            break
        if lr_func is not None:
            for g in opt.param_groups:
                g['lr'] = lr_func(epoch)
    if checkpoint is not None:
        model.load_state_dict(checkpoint)

    # import inspect
    # print(inspect.stack()[0][3])
    # print(inspect.stack()[1][3])
    return hist


class prior_mean_model_wrapper(torch.nn.Module):
    def __init__(self, prior_mean_model, decision_transformer, environmental_parameters):
        super().__init__()
        self.prior_mean_model = prior_mean_model
        self.transformer = decision_transformer
        if environmental_parameters is None:
            self.env = None
        else:
            self.env = torch.tensor(
                environmental_parameters, dtype=torch.float32)

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def forward(self, x_decision: Tensor) -> Tensor:
        shape = x_decision.shape
        n_batch = 1
        for n in shape[:-1]:
            n_batch *= n

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])
        with torch.no_grad():
            if self.env is None:
                return -self.prior_mean_model.forward(self.transformer(x_decision.view(n_batch, shape[-1]))).view(*shape[:-1])
            else:
                x_env = self.env.repeat(n_batch, 1)
                return -self.prior_mean_model.forward(self.transformer(x_decision.view(n_batch, shape[-1])), x_env).view(*shape[:-1])


class InteractiveGPBO(QObject):
    updateProgressBar = pyqtSignal(int)

    def __init__(self,
                 x,
                 y,
                 bounds,
                 batch_size=1,
                 acquisition_func=None,
                 acquisition_func_args=None,
                 # , "nonlinear_inequality_constraints":[]},
                 acquisition_optimize_options={
                     "num_restarts": 20, "raw_samples": 20},
                 scipy_minimize_options=None,
                 prior_mean_model=None,
                 prior_mean_model_env=None,
                 L2reg=0.0,
                 avoid_bounds_corners=True,
                 path="./log/",
                 tag=""
                 ):

        super().__init__()

        _, self.dim = x.shape

        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)

        assert type(batch_size) == int
        self.batch_size = batch_size

        self.bounds = np.array(bounds, dtype=np.float64)
        assert self.dim == len(self.bounds)
        for i in range(self.dim):
            assert self.bounds[i, 0] < self.bounds[i, 1]
            for x_ in x:
                assert x_[i] > self.bounds[i, 0] and x_[i] < self.bounds[i, 1]

        if acquisition_func in ["UCB", "UpperConfidenceBound", "qUpperConfidenceBound", "LCB", "LowerConfidenceBound"]:
            self.acquisition_func = qUpperConfidenceBound
        elif acquisition_func in ["KG", "qKnowledgeGradient", "KnowledgeGradient"]:
            self.acquisition_func = qKnowledgeGradient
        else:
            self.acquisition_func = acquisition_func

        if acquisition_func_args is None:
            if acquisition_func in ["UCB", "UpperConfidenceBound", "qUpperConfidenceBound", "LCB", "LowerConfidenceBound"]:
                acquisition_func_args = {"beta": 4.0}
            else:
                acquisition_func_args = {"num_fantasies": 64}
        self.acquisition_func_args = acquisition_func_args
        self.acquisition_optimize_options = acquisition_optimize_options
        self.scipy_minimize_options = scipy_minimize_options

        self.prior_mean_model = prior_mean_model
        if prior_mean_model is not None:
            for param in prior_mean_model.parameters():
                param.requires_grad = False
            self.prior_mean_model = prior_mean_model_wrapper(prior_mean_model,
                                                             decision_transformer=self.unnormalize,
                                                             environmental_parameters=prior_mean_model_env)

        self.L2reg = L2reg
        self.avoid_bounds_corners = avoid_bounds_corners

        if path[-1] != "/":
            path += "/"
        self.path = path
        self.tag = tag
        self.history = {'x': [],
                        'y': [],
                        'gp': [],
                        'acquisition': [],
                        'bounds': [],
                        'beta': [],
                        'x1': []}
        self.epoch = 0

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def normalize(self, x, bounds=None):
        if bounds is None:
            bounds = self.bounds
        diff = bounds[:, 1] - bounds[:, 0]
        out = 2 * (x-bounds[:, 0]) / diff - 1.0

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])
        return out

    def unnormalize(self, x, bounds=None):
        if bounds is None:
            bounds = self.bounds
        bounds = torch.tensor(bounds, dtype=torch.float64)
        diff = bounds[:, 1] - bounds[:, 0]
        out = 0.5 * (x.to(torch.float64) + 1.) * diff + bounds[:, 0]

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])
        return out.to(torch.float32)

    def step(self, x=None, y=None,
             batch_size=None,
             acquisition_func=None,
             acquisition_func_args=None,
             acquisition_optimize_options=None,
             scipy_minimize_options=None,
             X_pending=None,
             L2reg=None,
             log=True,
             remember_gp=True,
             write_gp_on_grid=False, grid_points_each_dim=25):

        if x is None:
            assert y is None
        else:
            x = np.array(x, dtype=np.float64)
        if y is None:
            assert x is None
            x = self.x
            y = self.y
        else:
            y = np.array(y, dtype=np.float64)
        xt = torch.tensor(self.normalize(x), dtype=torch.float32)
        # sign flip for minimization (instead of maximization)
        yt = torch.tensor(-y, dtype=torch.float32)

        L2reg = L2reg or self.L2reg
        if L2reg > 0.0:
            reg = L2reg * xt**2
            yt += torch.sum(reg, dim=1)[:, None]

        gp = SingleTaskGP(xt, yt, mean_module=self.prior_mean_model)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll, options=self.scipy_minimize_options)

        if acquisition_func is None:
            acquisition_func = self.acquisition_func
        if acquisition_func_args is None:
            acquisition_func_args = self.acquisition_func_args
        if batch_size is None:
            batch_size = self.batch_size
        if self.avoid_bounds_corners and self.prior_mean_model is None:
            corner_grid = [(-1, 1)]*self.dim
            corner_grid = np.meshgrid(*corner_grid)
            corner_grid = np.array(list(list(x.flat)
                                   for x in corner_grid), np.float32).T
            corner_grid = torch.tensor(corner_grid)
            acquisition_func_args["X_pending"] = corner_grid

        acquisition_optimize_options = acquisition_optimize_options or self.acquisition_optimize_options
        scipy_minimize_options = scipy_minimize_options or self.scipy_minimize_options

        acqu = acquisition_func(gp, **acquisition_func_args)
        xt1, yt1acqu = optimize_acqf(acqu,
                                     bounds=torch.stack(
                                         [-torch.ones(self.dim), torch.ones(self.dim)]),
                                     q=batch_size, **acquisition_optimize_options)  # ,

        x1 = self.unnormalize(xt1).detach().numpy()

        self.x = copy(x)
        self.y = copy(y)
        self.gp = gp
        self.acquisition = acqu
        self.history['x'].append(copy(self.x))
        self.history['y'].append(copy(self.y))
        self.history['bounds'].append(copy(self.bounds))
        self.history['x1'].append(copy(x1))
        self.epoch += 1
        if log:
            self.write_log(path=self.path, tag=self.tag)
        if remember_gp or write_gp_on_grid:
            self.history['gp'].append(gp)
            self.history['acquisition'].append(acqu)
            if write_gp_on_grid:
                self.write_gp_on_grid(
                    self.epoch, grid_points_each_dim=grid_points_each_dim)
        else:
            self.history['gp'].append(None)
            self.history['acquisition'].append(None)
                

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])
        return x1

    def update_exploration(self, x1, y1, x=None, y=None):
        if x is None:
            assert y is None
        if y is None:
            assert x is None
            x = self.x
            y = self.y

        x = np.vstack((x, x1))
        y = np.vstack((y, y1))
        self.x = x
        self.y = y

        self.history['y_min'] = np.array(
            [min(y[:i + 1]) for i in range(len(y))]).flatten()

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])
        return x, y

    def write_log(self,path="./log/",tag=""):
        if path[-1]!="/":
            path += "/"
        if not os.path.isdir(path):
            os.mkdir(path)
        tag = str(date.today())+"_"+tag
        if tag[-1] != "_":
            tag = tag+"_"
            
        if type(self.acquisition) == qUpperConfidenceBound:
            acquisition_type = 'UCB'
        elif type(self.acquisition) == qKnowledgeGradient:
            acquisition_type = 'KG'
            
        data = {'x':self.history['x'],
                'y':self.history['y'],
                'bounds':self.history['bounds'],
                'beta':self.history['beta'],
                'x1':self.history['x1'],
                'gp':[None]*self.epoch,
                'acquisition':[None]*self.epoch,
                'acquisition_type': acquisition_type,
                'acquisition_args': self.acquisition_func_args}
        pickle.dump(data,open(path+tag+"history.pickle","wb"))

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def load_from_log(self, fname=""):
        self.history = pickle.load(open(fname, "rb"))
        self.x = self.history['x'][-1]
        self.y = self.history['y'][-1]
        self.batch_size = len(self.history['x1'][-1])
        self.bounds = self.history['bounds'][-1]
        self.dim = len(self.bounds)
        self.epoch = len(self.history['x'])
        self.acquisition_func_args = self.history['acquisition_args']
        self.acquisition_optimize_options = {"num_restarts": 20, "raw_samples": 20}
        
        if self.history['acquisition_type'] in ["UCB", "UpperConfidenceBound", "qUpperConfidenceBound", "LCB", "LowerConfidenceBound"]:
            self.acquisition_func = qUpperConfidenceBound
        elif self.history['acquisition_type'] in ["KG", "qKnowledgeGradient", "KnowledgeGradient"]:
            self.acquisition_func = qKnowledgeGradient

        return self.x, self.y, self.bounds, self.batch_size, self.epoch, self.history['acquisition_type'], self.acquisition_func_args

    def write_gp_on_grid(self, epoch=None, path="./log/", tag="", grid_points_each_dim=25):
        if grid_points_each_dim**self.dim * self.dim > 2e6:
            print("Due to high-dimensionality and large number of grid point, GP data on grid will not be created nor saved into file. Try again with lower number of 'grid_points_each_dim'")
            return
        if path[-1] != "/":
            path += "/"
        if not os.path.isdir(path):
            os.mkdir(path)
        tag = str(date.today()) + "_" + tag
        if tag[-1] != "_":
            tag = tag+"_"
        if epoch is None:
            iiter = range(1, self.epoch + 1)
        else:
            iiter = [epoch]
        for i in iiter:
            if self.history['gp'][i-1] is None:
                print("gp at epoch " + str(i) +
                      " is not saved in memeory. GP data on grid will not be create and saved into file")
                continue
            linegrid = [np.linspace(self.history['bounds'][i - 1][d, 0],
                                    self.history['bounds'][i - 1][d, 1],
                                    grid_points_each_dim, dtype=np.float32) for d in range(self.dim)]
            x_grid = np.meshgrid(*linegrid)
            x_grid = np.array(list(list(x.flat)
                              for x in x_grid)).T.astype(np.float32)
            x_grid_t = torch.tensor(self.normalize(
                x_grid, self.history['bounds'][i-1]), dtype=torch.float32)
            with torch.no_grad():
                # decide batch size in multiple of "grid_points_each_dim"
                batch_size = 1
                for n in range(self.dim):
                    batch_size *= grid_points_each_dim
                    if batch_size*self.dim > 2e5:
                        batch_size = int(batch_size/grid_points_each_dim)
                        break
                n_batch = int(len(x_grid)/batch_size)
                # scan over grid points in chunk of batch_size
                y_mean = []
                y_acq = []
                for b in range(n_batch):
                    i1 = b*batch_size
                    i2 = i1 + batch_size
                    x_batch = x_grid_t[i1:i2]
                    y_pred = self.history['gp'][i - 1](x_batch)
                    y_mean.append(y_pred.mean.detach().numpy())
                    del y_pred
                    if "knowledge_gradient" in str(type(self.history['acquisition'][i - 1])):
                        model = self.history['gp'][i - 1]
                        acq = UpperConfidenceBound(model, beta=4)
                        y_acq.append(
                            acq(x_batch.view(-1, 1, self.dim)).detach().numpy())
                    else:
                        y_acq.append(
                            self.history['acquisition'][i - 1](x_batch.view(-1, 1, self.dim)).detach().numpy())
            data = {'x_grid': x_grid,
                    'y_mean': -np.concatenate(y_mean, axis=0),
                    'y_acq': -np.concatenate(y_acq, axis=0),
                    'x': self.history['x'][i - 1],
                    'y': self.history['y'][i - 1],
                    'x1': self.history['x1'][i - 1], }
            pickle.dump(data, open(
                path + tag + "GPlog_epoch" + str(i) + ".pickle", "wb"))

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def get_best(self):
        ymin = np.inf
        for i in range(self.epoch):
            if np.min(self.history['y'][i]) < ymin:
                imin = np.argmin(self.history['y'][i])
                xmin = self.history['x'][i][imin]
                ymin = self.history['y'][i][imin]

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])
        return xmin, ymin

    def plot_best(self, ax=None):
        y_best_hist = [np.min(self.history['y'][-1][:i + 1])
                       for i in range(len(self.history['y'][-1]))]

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(y_best_hist)
        ax.set_xlabel("evaluation budget")
        ax.set_ylabel("loss")

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])
        return ax

    def plot_aquisition_2D_projection(self, epoch=None,
                                      dim_xaxis=0, dim_yaxis=1,
                                      grid_points_each_dim=25,
                                      project_minimum=True,
                                      project_mean=False,
                                      fixed_values_for_each_dim=None,
                                      overdrive=False,
                                      ax=None, axes=None):
        '''
        fixed_values_for_each_dim: dict of key: dimension, val: value to fix for that dimension
        '''
        if grid_points_each_dim**(self.dim - 2)*(self.dim - 2) > 2e6 and (project_minimum or project_mean):
            print("Due to high-dimensionality and large number of grid point, plot_aquisition_2D_projection with 'project_minimum=True' or 'project_mean=True' cannot proceed. Try again with lower number of 'grid_points_each_dim' or use 'fixed_values_for_each_dim'")
            return
        if epoch is None:
            epoch = self.epoch

        if hasattr(self.history['acquisition'][epoch - 1], 'beta_prime'):
            beta = (self.history['acquisition']
                    [epoch - 1].beta_prime * 2 / np.pi)**2
        else:
            beta = 4
        acq = UpperConfidenceBound(self.history['gp'][epoch - 1], beta=beta)

        if self.dim > 2:
            batch_size = 1
            for n in range(self.dim-2):
                batch_size *= grid_points_each_dim

                if batch_size * self.dim > 2e4:
                    if overdrive or not (project_minimum or project_mean):
                        batch_size = int(batch_size / grid_points_each_dim)
                        print("starting projection plot...")
                        break
                    else:
                        raise RuntimeError(
                            "Aborting: due to high-dimensionality and large number of grid point, minimum or mean projection plot may take long time. Try to reduce 'grid_points_each_dim' or turn on 'overdrive' if long time waiting is OK'")
            n_batch = int(grid_points_each_dim**(self.dim - 2)/batch_size)

        linegrid = np.linspace(0, 1, grid_points_each_dim)
        x_grid = np.zeros(
            (grid_points_each_dim*grid_points_each_dim, self.dim))
        y_grid = np.zeros((grid_points_each_dim*grid_points_each_dim))

        n = 0
        for i in self.calculateProgress(range(grid_points_each_dim)):
            bounds_xaxis = self.history['bounds'][epoch - 1][dim_xaxis, :]
            for j in range(grid_points_each_dim):
                bounds_yaxis = self.history['bounds'][epoch - 1][dim_yaxis, :]
                x_grid[n, dim_xaxis] = linegrid[i] * \
                    (bounds_xaxis[1] - bounds_xaxis[0]) + bounds_xaxis[0]
                x_grid[n, dim_yaxis] = linegrid[j] * \
                    (bounds_yaxis[1] - bounds_yaxis[0]) + bounds_yaxis[0]
                if (project_minimum or project_mean) and self.dim > 2:
                    inner_grid = []
                    for d in range(self.dim):
                        if d == dim_xaxis:
                            inner_grid.append([x_grid[n, dim_xaxis]])
                        elif d == dim_yaxis:
                            inner_grid.append([x_grid[n, dim_yaxis]])
                        else:
                            inner_grid.append(np.linspace(self.history['bounds'][epoch - 1][d, 0],
                                                          self.history['bounds'][epoch - 1][d, 1],
                                                          grid_points_each_dim))

                    inner_grid = np.meshgrid(*inner_grid)
                    inner_grid = np.array(list(list(x.flat)
                                          for x in inner_grid), np.float32).T
                    inner_grid_n = torch.tensor(self.normalize(
                        inner_grid, self.history['bounds'][epoch - 1]), dtype=torch.float32)

                    y_acq = []
                    with torch.no_grad():
                        for b in range(n_batch):
                            i1 = b*batch_size
                            i2 = i1 + batch_size
                            x_batch = inner_grid_n[i1:i2]
                            y_acq.append(
                                acq(x_batch.view(-1, 1, self.dim)).detach().numpy())

                    y_acq = -np.concatenate(y_acq, axis=0)

                    if project_minimum:
                        imin = np.argmin(y_acq)
                        y_grid[n] = y_acq[imin]
                        x_grid[n, :] = inner_grid[imin]
                    elif project_mean:
                        y_grid[n] = np.mean(y_acq)
                n += 1

        if (not (project_minimum or project_mean)) or self.dim == 2:
            if fixed_values_for_each_dim is not None:
                for dim, val in fixed_values_for_each_dim.items():
                    x_grid[:, dim] = val

            x_grid_n = torch.tensor(self.normalize(
                x_grid, self.history['bounds'][epoch - 1]), dtype=torch.float32)
            y_grid = -acq(x_grid_n.view(-1, 1, self.dim)).detach().numpy()

        x = self.history['x'][epoch - 1]
        x1 = self.history['x1'][epoch - 1]

        if ax is None and axes is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        elif axes:
            for ax in axes:
                data = ax.tricontourf(
                    x_grid[:, dim_xaxis], x_grid[:, dim_yaxis], y_grid, levels=64, cmap="viridis")
                if self.dim > 2:
                    tag = "projected "
                else:
                    tag = ""
                ax.scatter(x[:, dim_xaxis], x[:, dim_yaxis],
                           c="b", label=tag + "training data")
                ax.scatter(x1[:, dim_xaxis], x1[:, dim_yaxis],
                           c="r", label=tag + "candidate")
                
                cbar = addColorBar(ax, data, y_grid)
                
            return axes

        data = ax.tricontourf(x_grid[:, dim_xaxis], x_grid[:,
                       dim_yaxis], y_grid, levels=64, cmap="viridis")
        if self.dim > 2:
            tag = "projected "
        else:
            tag = ""
        ax.scatter(x[:, dim_xaxis], x[:, dim_yaxis],
                   c="b", label=tag + "training data")
        ax.scatter(x1[:, dim_xaxis], x1[:, dim_yaxis],
                   c="r", label=tag + "candidate")
        
        addColorBar(ax, data, y_grid)
        
        return ax
        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def plot_GPmean_2D_projection(self, epoch=None,
                                  dim_xaxis=0, dim_yaxis=1,
                                  grid_points_each_dim=25,
                                  project_minimum=True,
                                  project_mean=False,
                                  fixed_values_for_each_dim=None,
                                  overdrive=False,
                                  ax=None, axes=None):
        '''
        fixed_values_for_each_dim: dict of key: dimension, val: value to fix for that dimension
        '''

        if epoch is None:
            epoch = self.epoch

        if self.dim > 2:
            batch_size = 1
            for n in range(self.dim-2):
                batch_size *= grid_points_each_dim

                if batch_size*self.dim > 2e4:
                    if overdrive or not (project_minimum or project_mean):
                        batch_size = int(batch_size / grid_points_each_dim)
                        print("starting projection plot...")
                        break
                    else:
                        raise RuntimeError(
                            "Aborting: due to high-dimensionality and large number of grid point, minimum or mean projection plot may take long time. Try to reduce 'grid_points_each_dim' or turn on 'overdrive' if long time waiting is OK'")
            n_batch = int(grid_points_each_dim**(self.dim-2) / batch_size)

        gp = self.history['gp'][epoch - 1]

        linegrid = np.linspace(0, 1, grid_points_each_dim)
        x_grid = np.zeros(
            (grid_points_each_dim * grid_points_each_dim, self.dim))
        y_grid = np.zeros((grid_points_each_dim * grid_points_each_dim))
        n = 0
        for i in self.calculateProgress(range(grid_points_each_dim)):
            bounds_xaxis = self.history['bounds'][epoch - 1][dim_xaxis, :]
            for j in range(grid_points_each_dim):
                bounds_yaxis = self.history['bounds'][epoch - 1][dim_yaxis, :]
                x_grid[n, dim_xaxis] = linegrid[i] * \
                    (bounds_xaxis[1] - bounds_xaxis[0]) + bounds_xaxis[0]
                x_grid[n, dim_yaxis] = linegrid[j] * \
                    (bounds_yaxis[1] - bounds_yaxis[0]) + bounds_yaxis[0]
                if (project_minimum or project_mean) and self.dim > 2:
                    inner_grid = []
                    for d in range(self.dim):
                        if d == dim_xaxis:
                            inner_grid.append([x_grid[n, dim_xaxis]])
                        elif d == dim_yaxis:
                            inner_grid.append([x_grid[n, dim_yaxis]])
                        else:
                            inner_grid.append(np.linspace(self.history['bounds'][epoch - 1][d, 0],
                                                          self.history['bounds'][epoch - 1][d, 1],
                                                          grid_points_each_dim))

                    inner_grid = np.meshgrid(*inner_grid)
                    inner_grid = np.array(list(list(x.flat)
                                          for x in inner_grid), np.float32).T
                    inner_grid_n = torch.tensor(self.normalize(
                        inner_grid, self.history['bounds'][epoch - 1]), dtype=torch.float32)

                    y_mean = []
                    with torch.no_grad():
                        for b in range(n_batch):
                            i1 = b*batch_size
                            i2 = i1 + batch_size
                            x_batch = inner_grid_n[i1:i2]
                            y_posterior = gp(x_batch)
                            y_mean.append(-y_posterior.mean.detach().numpy())
                    y_mean = np.concatenate(y_mean, axis=0)

                    if project_minimum:
                        imin = np.argmin(y_mean)
                        y_grid[n] = y_mean[imin]
                        x_grid[n, :] = inner_grid[imin]
                    elif project_mean:
                        y_grid[n] = np.mean(y_mean)

                n += 1

        if (not (project_minimum or project_mean)) or self.dim == 2:
            if fixed_values_for_each_dim is not None:
                for dim, val in fixed_values_for_each_dim.items():
                    x_grid[:, dim] = val

            x_grid_n = torch.tensor(self.normalize(
                x_grid, self.history['bounds'][epoch - 1]), dtype=torch.float32)
            y_posterior = gp(x_grid_n)
            y_grid = -y_posterior.mean.detach().numpy()

        x = self.history['x'][epoch - 1]
        x1 = self.history['x1'][epoch - 1]

        if ax is None and axes is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        elif axes:
            for ax in axes:
                data = ax.tricontourf(
                    x_grid[:, dim_xaxis], x_grid[:, dim_yaxis], y_grid, levels=64, cmap="viridis")
                if self.dim > 2:
                    tag = "projected "
                else:
                    tag = ""
                ax.scatter(x[:, dim_xaxis], x[:, dim_yaxis],
                           c="b", label=tag + "training data")
                ax.scatter(x1[:, dim_xaxis], x1[:, dim_yaxis],
                           c="r", label=tag + "candidate")

                addColorBar(ax, data, y_grid)
                
            return axes

        data = ax.tricontourf(x_grid[:, dim_xaxis], x_grid[:,
                       dim_yaxis], y_grid, levels=64, cmap="viridis")
        if self.dim > 2:
            tag = "projected "
        else:
            tag = ""
        ax.scatter(x[:, dim_xaxis], x[:, dim_yaxis],
                   c="b", label=tag + "training data")
        ax.scatter(x1[:, dim_xaxis], x1[:, dim_yaxis],
                   c="r", label=tag + "candidate")
        
        addColorBar(ax, data, y_grid)
        
        return ax
        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def plot_obj_history(self, plot_best_only=False, ax=None, axes=None):
        y_best_hist = [np.min(self.history['y'][-1][:i + 1])
                       for i in range(len(self.history['y'][-1]))]

        if ax is None and axes is None:
            _, ax = plt.subplots(figsize=(4, 3))
        elif axes:
            for ax in axes:
                ax.plot(y_best_hist, label='best obj')
                if not plot_best_only:

                    def has_twin(ax):
                        for other_ax in ax.figure.axes:
                            if other_ax is ax:
                                continue
                            if other_ax.bbox.bounds == ax.bbox.bounds:
                                return True
                        return False

                    if not has_twin(ax):
                        ax1 = ax.twinx()
                    else:
                        ax1 = ax.get_shared_x_axes().get_siblings(ax)[0]

                    ax1.plot(self.history['y'][-1], c='C1', label='obj')
                    ax1.set_ylabel('obj history')
                    # ax1.set_xbound(0, len(y_best_hist))
                ax.set_xlabel("evaluation budget")
                ax.set_ylabel("obj minimum")
                # ax.set_xbound(lower=0, upper=len(y_best_hist))
                ax.set_aspect('auto')
                ax.set_xlim([0, len(y_best_hist)])
                ax.set_xbound(lower=0.0, upper=len(y_best_hist))
            return axes

        ax.plot(y_best_hist, label='best obj')
        if not plot_best_only:
            ax1 = ax.twinx()
            ax1.plot(self.history['y'][-1], label='obj')
            ax1.set_ylabel('obj history')
        ax.set_xlabel("evaluation budget")
        ax.set_ylabel("obj minimum")
        ax.set_aspect('auto')
        ax.set_xlim([0, len(y_best_hist)])
        ax.set_xbound(lower=0.0, upper=len(y_best_hist))
        return ax

    def calculateProgress(self, it, size=40):  # Python3.6+
        count = len(it)

        self.updateProgressBar.emit(0)
        for i, item in enumerate(it):
            yield item
            self.updateProgressBar.emit(int(size * (i + 1) / count))
