import os
import pickle
import sys
import threading
import time
import json
import warnings
from copy import deepcopy as copy
from datetime import date
from typing import (Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type,
                    Union)
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5.QtCore import QObject, pyqtSignal
from torch import Tensor
from PyQt5.QtWidgets import QWidget, QMessageBox
import components as cmp
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

warnings.filterwarnings("ignore")

dtype = torch.double

#############
# Functions #
#############

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
        if time.time()-t0 > maxtime:
            break
        if lr_func is not None:
            for g in opt.param_groups:
                g['lr'] = lr_func(epoch)
    if checkpoint is not None:
        model.load_state_dict(checkpoint)
    return hist

def addColorBar(ax, data, y_grid, main_window):        
    cbars = main_window.colorbars
    size_logic = lambda size: size - 2 if isinstance(ax.figure.canvas, cmp.plot.PreviewCanvas) else size
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    max_ygrid = max(y_grid)
    min_ygrid = min(y_grid)
    range_ygrid = max_ygrid - min_ygrid
    num_ticks = 1 if np.isclose(max_ygrid, min_ygrid) else 5
    tick_difference = range_ygrid / (num_ticks - 1) if num_ticks > 1 else 0
    ticks = [max_ygrid]
    for i in range(num_ticks):
        ticks.append(max_ygrid - i * tick_difference)
    cbar = ax.get_figure().colorbar(data, cax=cax, ticks=ticks)
    cbar.ax.tick_params(axis="both", labelsize=size_logic(10))
    
    if ax in cbars.keys():
        cbars[ax].remove()
    cbars[ax] = cbar

        
                
def calcDistance(pnt_A, pnt_B, weights=None):
    distances = []
    for i in range(len(pnt_A)):
        coord_A = pnt_A[i]
        coord_B = pnt_B[i]
        if weights:
            distances.append(abs(coord_A - coord_B) * weights[i])
        else:
            distances.append(abs(coord_A - coord_B))
    return max(distances)

def findBestPoint(prev_pnt, eval_pnts, weights=None):
    # calculating distances
    eval_pnt_distances = []
    for eval_pnt in eval_pnts:
            distance = calcDistance(prev_pnt, eval_pnt, weights)
            eval_pnt_distances.append(distance)
            
    pnt_index = eval_pnt_distances.index(min(eval_pnt_distances))
    return eval_pnts[pnt_index], pnt_index

def orderByDistance(prev_pnt, eval_pnts, weights=None):
    ordered_pnts = []

    if len(eval_pnts[0]) != len(weights):
        weights = []

    while eval_pnts.size != 0:
        best_pnt, best_pnt_index = findBestPoint(prev_pnt, eval_pnts, weights)
        ordered_pnts.append(best_pnt)
        eval_pnts = np.delete(eval_pnts, best_pnt_index, 0)
    return np.array(ordered_pnts)

###########
# Classes #
###########

##############
# Subclasses #
##############

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
            n_decision+n_env, list_of_nodes[0]), activation]
        for i in range(1, len(list_of_nodes)-1):
            self.seq += [torch.nn.Linear(list_of_nodes[i-1],
                                         list_of_nodes[i]), activation]
        self.seq += [torch.nn.Linear(list_of_nodes[-2], list_of_nodes[-1])]
        self.seq = torch.nn.Sequential(*self.seq)

    def forward(self, x_decision, x_env=None):
        if self.n_known_env > 0:
            x = torch.cat((x_decision, x_env), 1)
        else:
            x = x_decision
        if self.bounds is not None:
            x = self.normalize(x)
        return self.seq(x)

    def normalize(self, x):
        with torch.no_grad():
            diff = self.bounds[:, 1] - self.bounds[:, 0]
            out = 2*(x.to(torch.float64)-self.bounds[:, 0])/diff - 1.0
        return out.to(torch.float32)


mse = torch.nn.MSELoss()

class prior_mean_model_wrapper(torch.nn.Module):
    def __init__(self, prior_mean_model, decision_transformer, **kwargs):
        super().__init__()

        self.prior_mean_model = prior_mean_model
        if 'keras' in str(type(prior_mean_model)):
            def model(x):
                if type(x) is torch.Tensor:
                    x = x.detach().numpy()
                y = prior_mean_model(x)
                return y.numpy()
            self.prior_mean_model = model

        self.transformer = decision_transformer
        if 'env' in kwargs.keys():
            self.env = kwargs['env']
            del kwargs['env']
        else:
            self.env = None
        self.kwargs = kwargs

    def forward(self, x_decision: Tensor) -> Tensor:
        shape = x_decision.shape
        n_batch = 1
        for n in shape[:-1]:
            n_batch *= n
        with torch.no_grad():

            if self.env is None:
                return -torch.tensor(
                    self.prior_mean_model(
                        self.transformer(x_decision.view(n_batch, shape[-1])), **self.kwargs), dtype=x_decision.dtype).view(*shape[:-1])
            else:
                x_env = torch.tensor(
                    self.env, dtype=x_decision.dtype).repeat(n_batch, 1)

                return -torch.tensor(
                    self.prior_mean_model(
                        self.transformer(x_decision.view(n_batch, shape[-1])), x_env, **self.kwargs), dtype=x_decision.dtype).view(*shape[:-1])


class interactiveGPBO():
    def __init__(self,
                 x=None,
                 y=None,
                 yvar=None,
                 bounds=None,
                 noise_constraint=None,
                 load_log_fname='',
                 batch_size=1,
                 acquisition_func=None,
                 acquisition_func_args=None,
                 # , "nonlinear_inequality_constraints":[]},
                 acquisition_optimize_options={
                     "num_restarts": 20, "raw_samples": 20},
                 scipy_minimize_options=None,
                 prior_mean_model=None,
                 prior_mean_model_env=None,
                 prior_mean_model_kwargs={},
                 avoid_bounds_corners=True,
                 path="./log/",
                 tag=""
                 ):

        if load_log_fname != '':
            self.load_from_log(load_log_fname,
                               acquisition_func=acquisition_func,
                               acquisition_func_args=acquisition_func_args,
                               acquisition_optimize_options=acquisition_optimize_options,
                               scipy_minimize_options=scipy_minimize_options,
                               prior_mean_model=prior_mean_model,
                               prior_mean_model_env=prior_mean_model_env,
                               prior_mean_model_kwargs=prior_mean_model_kwargs,
                               path=path,
                               tag=tag)
            return

        self.dim = len(bounds)
        self.bounds = np.array(bounds, dtype=np.float64)

        assert x is not None
        assert y is not None
        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        print(self.x, self.y)

        _, dim = self.x.shape
        assert self.dim == dim
        
        if yvar is None:
            self.yvar = None
        else:
            self.yvar = np.array(yvar, dtype=np.float64)
        self.noise_constraint = noise_constraint

        for i in range(self.dim):
            assert self.bounds[i, 0] < self.bounds[i, 1]

        if acquisition_func in ["UCB", "UpperConfidenceBound", "qUpperConfidenceBound", "LCB", "LowerConfidenceBound"] or acquisition_func is None:
            self.acquisition_func = qUpperConfidenceBound
        elif acquisition_func in ["KG", "qKnowledgeGradient", "KnowledgeGradient"]:
            self.acquisition_func = qKnowledgeGradient
        else:
            self.acquisition_func = acquisition_func

        if acquisition_func_args is None:
            if acquisition_func in ["UCB", "UpperConfidenceBound", "qUpperConfidenceBound", "LCB", "LowerConfidenceBound"]:
                acquisition_func_args = {"beta": 2.0}
            else:
                acquisition_func_args = {"num_fantasies": 64}
        self.acquisition_func_args = acquisition_func_args
        self.acquisition_optimize_options = acquisition_optimize_options
        self.scipy_minimize_options = scipy_minimize_options

        assert type(batch_size) == int
        self.batch_size = batch_size

        if prior_mean_model is None:
            self.prior_mean_model = None
        else:
            if type(prior_mean_model) is torch.nn.Module:
                for param in prior_mean_model.parameters():
                    param.requires_grad = False
            if prior_mean_model_env is not None:
                prior_mean_model_kwargs['env'] = prior_mean_model_env
            self.prior_mean_model = prior_mean_model_wrapper(prior_mean_model,
                                                             decision_transformer=self.unnormalize,
                                                             **prior_mean_model_kwargs)
        if path[-1] != "/":
            path += "/"
        self.path = path
        self.tag = tag
        self.history = []
        self.update_GP(recycleGP=False)

    def normalize(self, x, bounds=None):
        if bounds is None:
            bounds = self.bounds
        diff = bounds[:, 1] - bounds[:, 0]
        out = 2*(x-bounds[:, 0])/diff - 1.0
        return out

    def unnormalize(self, x, bounds=None):
        if bounds is None:
            bounds = self.bounds
        bounds = torch.tensor(bounds, dtype=torch.float64)
        diff = bounds[:, 1] - bounds[:, 0]
        out = 0.5*(x.to(torch.float64)+1.)*diff + bounds[:, 0]
        return out.to(torch.float32)

    def update_GP(self, x1=None, y1=None, y1var=None,
                  x=None, y=None, yvar=None,  # use x and y, not x1 and y1
                  recycleGP="Auto",
                  noise_constraint=None,
                  log=True):

        if x is None or y is None:
            x = self.x
            y = self.y
            yvar = self.yvar
        else:
            x = np.array(x, dtype=np.float64)
            y = np.array(y, dtype=np.float64)
            if yvar is not None:
                yvar = np.array(yvar, dtype=np.float64)
        if x1 is not None or y1 is not None:
            assert x1 is not None and y1 is not None
            x = np.vstack((x, x1))
            y = np.vstack((y, y1))
            if yvar is not None:
                assert y1var is not None
                yvar = np.vstack((yvar, y1var))
        if x is None:
            return
        self.x = x
        self.y = y
        self.yvar = yvar
        noise_constraint = noise_constraint or self.noise_constraint

        xt = torch.tensor(self.normalize(x), dtype=torch.float32)
        yt = torch.tensor(-y, dtype=torch.float32)  # negative for minimization

        if hasattr(self, 'gp'):
            state_dict = self.gp.state_dict()
        else:
            state_dict = None
        if yvar is not None:
            yvart = torch.tensor(yvar, dtype=torch.float32)
            self.gp = FixedNoiseGP(
                xt, yt, yvart, mean_module=self.prior_mean_model)
        elif noise_constraint is None:
            self.gp = SingleTaskGP(xt, yt, mean_module=self.prior_mean_model)
        else:
            interval = Interval(noise_constraint[0], noise_constraint[1])
            likelihood = GaussianLikelihood(noise_constraint=interval)
            self.gp = SingleTaskGP(xt, yt,
                                   mean_module=self.prior_mean_model,
                                   likelihood=likelihood)

        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        if recycleGP:
            if recycleGP == 'Auto' and len(yt) < 10*self.dim:
                pass
            elif state_dict is not None:
                try:
                    self.gp.load_state_dict(state_dict)
                except Exception as exc:
                    print(exc)
                    print(
                        "previous liklihood model is not consistent with new model. GP hyper-parmaeters are not loaded.")

        fit_gpytorch_model(mll, options=self.scipy_minimize_options)

        hist = {'x': x,
                'y': y,
                'gp': self.gp,
                'acquisition': None,
                'bounds': self.bounds,
                'X_pending': [],
                'x1': [],
                }
        self.history.append(hist)

        if log:
            self.write_log(path=self.path, tag=self.tag)

    def query_candidates(self, batch_size=None,
                         acquisition_func=None,
                         acquisition_func_args=None,
                         acquisition_optimize_options=None,
                         scipy_minimize_options=None,
                         X_pending=None,
                         log=True,
                         ):

        if acquisition_func is None:
            acquisition_func = self.acquisition_func
            print("*", acquisition_func)
        if acquisition_func_args is None:
            acquisition_func_args = self.acquisition_func_args
        if batch_size is None:
            batch_size = self.batch_size

        if X_pending is not None:
            X_pending_n = self.normalize(X_pending)

            X_pending_n = torch.tensor(X_pending_n, dtype=torch.float32)
            acquisition_func_args["X_pending"] = X_pending_n

        acquisition_optimize_options = acquisition_optimize_options or self.acquisition_optimize_options
        scipy_minimize_options = scipy_minimize_options or self.scipy_minimize_options

        self.acquisition = acquisition_func(self.gp, **acquisition_func_args)
        xt1, yt1acqu = optimize_acqf(self.acquisition,
                                     bounds=torch.stack(
                                         [-torch.ones(self.dim), torch.ones(self.dim)]),
                                     q=batch_size, **acquisition_optimize_options)

        x1 = self.unnormalize(xt1).detach().numpy()
        hist = self.history[-1]
        hist['acquisition'] = self.acquisition
        hist['X_pending'].append(X_pending)
        hist['x1'].append(x1)
        if log:
            self.write_log(path=self.path, tag=self.tag)

        return x1

    def write_log(self, path="./log/", tag=""):
        if path[-1] != "/":
            path += "/"
        if not os.path.isdir(path):
            os.mkdir(path)
        
        data = []
        for hist in self.history:
            hist_ = {}
            for key, val in hist.items():
                if key == 'gp':
                    hist_[key] = None
                elif key == 'acquisition':
                    hist_[key] = val
                    if val is not None and type(val) is not str:
                        _ = str(type(val))
                        hist_[key] = _[_.rfind('.')+1:-2]
                    else:
                        hist_[key] = 'UCB'
                else:
                    hist_[key] = val
            data.append(hist_)
            
        tag = str(date.today())+"_"+tag
        if tag[-1] != "_":
            tag = tag+"_"
        pickle.dump(data, open(path+tag+"history.pickle", "wb"))
            

    def load_from_log(self,
                      fname="",
                      acquisition_func=None,
                      acquisition_func_args=None,
                      acquisition_optimize_options={
                          "num_restarts": 20, "raw_samples": 20},
                      scipy_minimize_options=None,
                      prior_mean_model=None,
                      prior_mean_model_env=None,
                      prior_mean_model_kwargs={},
                      path="./log/",
                      tag=""):

        if fname == 'Auto':
            flist = np.sort(os.listdir('./log/'))[::-1]
            found = False
            for f in flist:
                if f[4] == '-' and f[7] == '-' and f[10:] == '_history.pickle':
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    'Auto search of recent log failed. Input log file manually')
            fname = './log/'+f

        if ".pickle" in fname:
            hist = pickle.load(open(fname, "rb"))
        else:
            hist = json.load(open(fname, "r"))
            for i, element in enumerate(hist):
                for k, v in element.items():
                    if type(v) is list and k != "X_pending" and k != "x1":
                        hist[i][k] = np.array(v, np.float64)
            print(hist)
                    

        self.x = hist[-1]['x']
        self.y = hist[-1]['y']
        self.bounds = hist[-1]['bounds']
        try:
            self.batch_size = max(len(hist[-1]['x1'][-1]), 1)
        except Exception as exc:
            print(exc)
            self.batch_size = 1
        if acquisition_func is None:
            acquisition_func = hist[-1]['acquisition']

        # Modified copy of __init__() begins
        # ----------------------------------------------------------------------
        self.dim = len(self.bounds)
        self.bounds = np.array(self.bounds, dtype=np.float64)

        assert self.x is not None
        assert self.y is not None

        _, dim = np.array(self.x).shape
        assert self.dim == dim
        self.x = np.array(self.x, dtype=np.float64)
        self.y = np.array(self.y, dtype=np.float64)
        # if yvar is None:
        self.yvar = None
        # else:
        #     self.yvar = np.array(yvar, dtype=np.float64)
        self.noise_constraint = None

        for i in range(self.dim):
            assert self.bounds[i, 0] < self.bounds[i, 1]

        print(acquisition_func, "!")
        if acquisition_func in ["UCB", "UpperConfidenceBound", "qUpperConfidenceBound", "LCB", "LowerConfidenceBound"] or acquisition_func is None:
            print("success")
            self.acquisition_func = qUpperConfidenceBound
        elif acquisition_func in ["KG", "qKnowledgeGradient", "KnowledgeGradient"]:
            self.acquisition_func = qKnowledgeGradient
        else:
            self.acquisition_func = acquisition_func

        if acquisition_func_args is None:
            if acquisition_func in ["UCB", "UpperConfidenceBound", "qUpperConfidenceBound", "LCB", "LowerConfidenceBound"]:
                acquisition_func_args = {"beta": 2.0}
            else:
                acquisition_func_args = {"num_fantasies": 64}
        self.acquisition_func_args = acquisition_func_args
        self.acquisition_optimize_options = acquisition_optimize_options
        self.scipy_minimize_options = scipy_minimize_options

        if prior_mean_model is None:
            self.prior_mean_model = None
        else:
            if type(prior_mean_model) is torch.nn.Module:
                for param in prior_mean_model.parameters():
                    param.requires_grad = False
            if prior_mean_model_env is not None:
                prior_mean_model_kwargs['env'] = prior_mean_model_env
            self.prior_mean_model = prior_mean_model_wrapper(prior_mean_model,
                                                             decision_transformer=self.unnormalize,
                                                             **prior_mean_model_kwargs)
        if path[-1] != "/":
            path += "/"
        self.path = path
        self.tag = tag
        self.history = []
        self.update_GP(recycleGP=False)
        # ----------------------------------------------------------------------
        # Modified copy of __init__() ends

        self.history = hist
        self.history[-1]['gp'] = self.gp
        self.history[-1]['acquisition'] = self.acquisition_func

    def get_best(self):
        ymin = np.inf
        for i in range(len(self.history)):
            if np.min(self.history[i]['y']) < ymin:
                imin = np.argmin(self.history[i]['y'])
                xmin = self.history[i]['x'][imin]
                ymin = self.history[i]['y'][imin]
        return xmin, ymin

    def plot_best(self, axes:list=None,ax=None):

        y_best_hist = [np.min(self.history[-1]['y'][:i+1])
                       for i in range(len(self.history[-1]['y']))]

        if ax:
            axes = [ax]
        elif axes is None:
            fig, ax = plt.subplots(figsize=(4, 3))
            axes = [ax]
        for ax in axes:
            ax.plot(y_best_hist)
            ax.set_xlabel("evaluation budget")
            ax.set_ylabel("obj")
        return axes


    def plot_func_projection(self, func, bounds, x=None, dim_xaxis=0, dim_yaxis=1, 
                             grid_points_each_dim=25, project_minimum=True, 
                             project_mean=False, fixed_values_for_each_dim=None, 
                             overdrive=False):
        '''
        fixed_values_for_each_dim: dict of key: dimension, val: value to fix for that dimension
        '''
        dim = len(bounds)
        if dim > 2:
            batch_size = 1
            for n in range(dim-2):
                batch_size *= grid_points_each_dim
                if batch_size*dim > 2e4:
                    if overdrive or not (project_minimum or project_mean):
                        batch_size = int(batch_size/grid_points_each_dim)
                        print("starting projection plot...")
                        break
                    else:
                        raise RuntimeError(
                            "Aborting: due to high-dimensionality and large number of grid point, minimum or mean projection plot may take long time. Try to reduce 'grid_points_each_dim' or turn on 'overdrive' if long time waiting is OK'")
            n_batch = int(grid_points_each_dim**(dim-2)/batch_size)

        gp = func
        linegrid = np.linspace(0, 1, grid_points_each_dim)
        x_grid = np.zeros((grid_points_each_dim*grid_points_each_dim, dim))
        y_grid = np.zeros((grid_points_each_dim*grid_points_each_dim))
        n = 0
        for i in self.calculateProgress(range(grid_points_each_dim)):
            bounds_xaxis = bounds[dim_xaxis, :]
            for j in range(grid_points_each_dim):
                bounds_yaxis = bounds[dim_yaxis, :]
                x_grid[n, dim_xaxis] = linegrid[i] * \
                    (bounds_xaxis[1]-bounds_xaxis[0])+bounds_xaxis[0]
                x_grid[n, dim_yaxis] = linegrid[j] * \
                    (bounds_yaxis[1]-bounds_yaxis[0])+bounds_yaxis[0]
                if (project_minimum or project_mean) and dim > 2:
                    inner_grid = []
                    for d in range(dim):
                        if d == dim_xaxis:
                            inner_grid.append([x_grid[n, dim_xaxis]])
                        elif d == dim_yaxis:
                            inner_grid.append([x_grid[n, dim_yaxis]])
                        else:
                            inner_grid.append(np.linspace(bounds[d, 0],
                                                          bounds[d, 1],
                                                          grid_points_each_dim))

                    inner_grid = np.meshgrid(*inner_grid)
                    inner_grid = np.array(list(list(x.flat)
                                          for x in inner_grid), np.float32).T

                    y_mean = []
                    for b in range(n_batch):
                        i1 = b*batch_size
                        i2 = i1 + batch_size
                        x_batch = inner_grid[i1:i2]
                        y_mean.append(gp(x_batch))
                    y_mean = np.concatenate(y_mean, axis=0)
                    if project_minimum:
                        imin = np.argmin(y_mean)
                        y_grid[n] = y_mean[imin]
                        x_grid[n, :] = inner_grid[imin]
                    elif project_mean:
                        y_grid[n] = np.mean(y_mean)
                n += 1

        if (not (project_minimum or project_mean)) or dim == 2:
            if fixed_values_for_each_dim is not None:
                for dim, val in fixed_values_for_each_dim.items():
                    x_grid[:, dim] = val

            y_grid = gp(x_grid)
            
        if ax:
            axes = [ax]
        elif axes is None:
            fig, ax = plt.subplots(figsize=(4, 4))
            axes = [ax]
        for ax in axes:
            cs = ax.tricontourf(
                x_grid[:, dim_xaxis], x_grid[:, dim_yaxis], y_grid, levels=64, cmap="viridis")
            if dim > 2:
                tag = "projected "
            else:
                tag = ""
            if x is not None:
                ax.scatter(x[:, dim_xaxis], x[:, dim_yaxis],
                        c="b", label=tag+" data")
                
    def epoch(self):
        return len(self.history) - 1