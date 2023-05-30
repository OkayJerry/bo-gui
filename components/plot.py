from typing import List, Tuple, Union, NoReturn, Generator

import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5.QtCore import pyqtSignal


class BaseCanvas(FigureCanvasQTAgg):
    progressUpdate = pyqtSignal(int)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.acquisition_ax: Axes = None
        self.posterior_ax: Axes = None
        self.obj_history_ax: Axes = None
        self.colorbars: dict = {}
    def _get_obj_twinx(self) -> Union[Axes, None]:
        """
        Retrieves the twin axis of `self.obj_history_ax`.

        Returns:
            Union[Axes, None]: `Axes` if found, `None` if not.
        """
        shared_axes = self.obj_history_ax.get_shared_x_axes()
        siblings = shared_axes.get_siblings(self.obj_history_ax)
        
        if siblings:
            return siblings[0]
    def _handle_colorbar(self, ax, data, y_grid) -> None:
        size_logic = lambda size: size - 2 if isinstance(ax.figure.canvas, MainCanvas) else size
        
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
        
        if ax in self.colorbars.keys():
            self.colorbars[ax].remove()
        self.colorbars[ax] = cbar
    def _calculate_progress(self, it, size=40) -> Generator:  # Python3.6+
        count = len(it)

        self.progressUpdate.emit(0)
        for i, item in enumerate(it):
            yield item
            self.progressUpdate.emit(int(size * (i + 1) / count))
    def get_axes(self) -> List[Axes]:
        axes = []

        if self.acquisition_ax:
            axes.append(self.acquisition_ax)
        if self.posterior_ax:
            axes.append(self.posterior_ax)
        if self.obj_history_ax:
            axes.append(self.obj_history_ax)
            twinx = self._get_obj_twinx()
            if twinx:
                axes.append(twinx)

        return axes
    def clear(self) -> None:
        """Clears all axes on the canvas.

        Raises:
            ValueError: If any axes do not exist.
        """
        if not self.acquisition_ax or not self.posterior_ax or not self.obj_history_ax:
            raise ValueError("One or more of the required axes are missing.")
            
        for ax in self.get_axes():
            ax.clear()
    def hide_axes(self) -> None:
        """
        Hides all axes.
        """
        for ax in self.get_axes():
            if ax.get_visible():
                ax.set_visible(False)
    def plot_acquisition(self, GPBO, epoch=None, i_query=None, beta=None, dim_xaxis=0, dim_yaxis=1, project_minimum=True, project_mean=False, fixed_values_for_each_dim=None, overdrive=False, grid_points_each_dim=25, axes=None) -> None:
        '''
        fixed_values_for_each_dim: dict of key: dimension, val: value to fix for that dimension
        '''
        if not self.acquisition_ax and not axes:
            raise ValueError("No axes provided and `self.acquisition_ax` is `None`.")
        
        if grid_points_each_dim**(GPBO.dim-2)*(GPBO.dim-2) > 2e6 and (project_minimum or project_mean):
            print("Due to high-dimensionality and large number of grid point, plot_aquisition_2D_projection with 'project_minimum=True' or 'project_mean=True' cannot proceed. Try again with lower number of 'grid_points_each_dim' or use 'fixed_values_for_each_dim'")
            return
        if epoch is None:
            epoch = len(GPBO.history)-1

        if beta is None:
            if hasattr(GPBO.history[epoch]['acquisition'], 'beta_prime'):
                beta = (GPBO.history[epoch]
                        ['acquisition'].beta_prime*2/np.pi)**2
            else:
                beta = 2

        gp = GPBO.history[epoch]['gp']
        if gp is None:
            if overdrive:
                raise RuntimeError("acquisition function at epoch "+str(epoch) +
                                " is not saved into memory. Turn on 'overdrive' to proceed.")
            else:
                print("acquisition function at epoch "+str(epoch) +
                    " is not saved into memory. Trying to restore based on training data. Restoration will not be exactly same.")
                x = GPBO.history[epoch]['x']
                y = GPBO.history[epoch]['y']
                xt = torch.tensor(GPBO.normalize(x), dtype=torch.float32)
                # sign flip for minimization (instead of maximization)
                yt = torch.tensor(-y, dtype=torch.float32)
                gp = SingleTaskGP(xt, yt, mean_module=GPBO.prior_mean_model)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll, options=GPBO.scipy_minimize_options)
        acq = UpperConfidenceBound(gp, beta=beta)

        if GPBO.dim > 2:
            batch_size = 1
            for n in range(GPBO.dim-2):
                batch_size *= grid_points_each_dim

                if batch_size*GPBO.dim > 2e4:
                    if overdrive or not (project_minimum or project_mean):
                        batch_size = int(batch_size/grid_points_each_dim)
                        print("starting projection plot...")
                        break
                    else:
                        raise RuntimeError(
                            "Aborting: due to high-dimensionality and large number of grid point, minimum or mean projection plot may take long time. Try to reduce 'grid_points_each_dim' or turn on 'overdrive' if long time waiting is OK'")
            n_batch = int(grid_points_each_dim**(GPBO.dim-2)/batch_size)

        linegrid = np.linspace(0, 1, grid_points_each_dim)
        x_grid = np.zeros(
            (grid_points_each_dim*grid_points_each_dim, GPBO.dim))
        y_grid = np.zeros((grid_points_each_dim*grid_points_each_dim))

        n = 0
        for i in self._calculate_progress(range(grid_points_each_dim)):
            bounds_xaxis = GPBO.history[epoch]['bounds'][dim_xaxis, :]
            for j in range(grid_points_each_dim):
                bounds_yaxis = GPBO.history[epoch]['bounds'][dim_yaxis, :]
                x_grid[n, dim_xaxis] = linegrid[i] * \
                    (bounds_xaxis[1]-bounds_xaxis[0])+bounds_xaxis[0]
                x_grid[n, dim_yaxis] = linegrid[j] * \
                    (bounds_yaxis[1]-bounds_yaxis[0])+bounds_yaxis[0]
                if (project_minimum or project_mean) and GPBO.dim > 2:
                    inner_grid = []
                    for d in range(GPBO.dim):
                        if d == dim_xaxis:
                            inner_grid.append([x_grid[n, dim_xaxis]])
                        elif d == dim_yaxis:
                            inner_grid.append([x_grid[n, dim_yaxis]])
                        else:
                            inner_grid.append(np.linspace(GPBO.history[epoch]['bounds'][d, 0],
                                                        GPBO.history[epoch]['bounds'][d, 1],
                                                        grid_points_each_dim))

                    inner_grid = np.meshgrid(*inner_grid)
                    inner_grid = np.array(list(list(x.flat)
                                        for x in inner_grid), np.float32).T
                    inner_grid_n = torch.tensor(GPBO.normalize(
                        inner_grid, GPBO.history[epoch]['bounds']), dtype=torch.float32)

                    y_acq = []
                    with torch.no_grad():
                        for b in range(n_batch):
                            i1 = b*batch_size
                            i2 = i1 + batch_size
                            x_batch = inner_grid_n[i1:i2]
                            y_acq.append(
                                acq(x_batch.view(-1, 1, GPBO.dim)).detach().numpy())

                    y_acq = -np.concatenate(y_acq, axis=0)
                    if project_minimum:
                        imin = np.argmin(y_acq)
                        y_grid[n] = y_acq[imin]
                        x_grid[n, :] = inner_grid[imin]
                    elif project_mean:
                        y_grid[n] = np.mean(y_acq)
                n += 1

        if (not (project_minimum or project_mean)) or GPBO.dim == 2:
            if fixed_values_for_each_dim is not None:
                for dim, val in fixed_values_for_each_dim.items():
                    x_grid[:, dim] = val

            x_grid_n = torch.tensor(GPBO.normalize(
                x_grid, GPBO.history[epoch]['bounds']), dtype=torch.float32)
            y_grid = -acq(x_grid_n.view(-1, 1, GPBO.dim)).detach().numpy()

        x = GPBO.history[epoch]['x']

        if i_query is None:
            x1 = GPBO.history[epoch]['x1']
            if len(x1) > 0:
                x1 = np.vstack(x1)
            else:
                x1 = None

            X_pending = []
            for xp_ in GPBO.history[epoch]['X_pending']:
                if xp_ is not None:
                    X_pending.append(xp_)
            if len(X_pending) > 0:
                X_pending = np.vstack(X_pending)
            else:
                X_pending = None
        else:
            if i_query >= len(GPBO.history[epoch]['x1']):
                raise ValueError(
                    'i_query exceed number of query in the epoch '+str(epoch))
            x1 = GPBO.history[epoch]['x1'][i_query]
            X_pending = GPBO.history[epoch]['X_pending'][i_query]

        if self.acquisition_ax:
            axes = [self.acquisition_ax]
            
        for ax in axes:
            data = ax.tricontourf(
                x_grid[:, dim_xaxis], x_grid[:, dim_yaxis], y_grid, levels=64, cmap="viridis")
            tag = ""
            ax.scatter(x[:, dim_xaxis], x[:, dim_yaxis],
                    c="b", alpha=0.7, label="training data")
            if x1 is not None:
                ax.scatter(x1[:, dim_xaxis], x1[:, dim_yaxis],
                        c="r", alpha=0.7, label="candidate")
            if X_pending is not None:
                X_pending = np.atleast_2d(X_pending)
                ax.scatter(X_pending[:, dim_xaxis], X_pending[:,
                        dim_yaxis], s=50, c="r", marker='x', label="pending")
            
            self._handle_colorbar(ax, data, y_grid)
    def plot_posterior(self, GPBO, epoch=None, i_query=None, dim_xaxis=0, dim_yaxis=1, project_minimum=True, project_mean=False, fixed_values_for_each_dim=None, overdrive=False, grid_points_each_dim=25, axes=None) -> None:
        '''
        fixed_values_for_each_dim: dict of key: dimension, val: value to fix for that dimension
        '''
        if not self.posterior_ax and not axes:
            raise ValueError("No axes provided and `self.acquisition_ax` is `None`.")

        if epoch is None:
            epoch = len(GPBO.history)-1

        if GPBO.dim > 2:
            batch_size = 1
            for n in range(GPBO.dim-2):
                batch_size *= grid_points_each_dim

                if batch_size*GPBO.dim > 2e4:
                    if overdrive or not (project_minimum or project_mean):
                        batch_size = int(batch_size/grid_points_each_dim)
                        print("starting projection plot...")
                        break
                    else:
                        raise RuntimeError(
                            "Aborting: due to high-dimensionality and large number of grid point, minimum or mean projection plot may take long time. Try to reduce 'grid_points_each_dim' or turn on 'overdrive' if long time waiting is OK'")
            n_batch = int(grid_points_each_dim**(GPBO.dim-2)/batch_size)

        gp = GPBO.history[epoch]['gp']
        if gp is None:
            if overdrive:
                raise RuntimeError(
                    "gp at epoch "+str(epoch)+" is not saved into memory. Turn on 'overdrive' to proceed.")
            else:
                print("gp function at epoch "+str(epoch) +
                      " is not saved into memory. Trying to restore based on training data. Restoration will not be exactly same.")
                x = GPBO.history[epoch]['x']
                y = GPBO.history[epoch]['y']
                xt = torch.tensor(GPBO.normalize(x), dtype=torch.float32)
                # sign flip for minimization (instead of maximization)
                yt = torch.tensor(-y, dtype=torch.float32)
                L2reg = GPBO.L2reg
                if L2reg > 0.0:
                    reg = L2reg*xt**2
                    yt += torch.sum(reg, dim=-1, keepdim=True)
                gp = SingleTaskGP(xt, yt, mean_module=GPBO.prior_mean_model)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll, options=GPBO.scipy_minimize_options)

        linegrid = np.linspace(0, 1, grid_points_each_dim)
        x_grid = np.zeros(
            (grid_points_each_dim*grid_points_each_dim, GPBO.dim))
        y_grid = np.zeros((grid_points_each_dim*grid_points_each_dim))
        n = 0
        for i in self._calculate_progress(range(grid_points_each_dim)):
            bounds_xaxis = GPBO.history[epoch]['bounds'][dim_xaxis, :]
            for j in range(grid_points_each_dim):
                bounds_yaxis = GPBO.history[epoch]['bounds'][dim_yaxis, :]
                x_grid[n, dim_xaxis] = linegrid[i] * \
                    (bounds_xaxis[1]-bounds_xaxis[0])+bounds_xaxis[0]
                x_grid[n, dim_yaxis] = linegrid[j] * \
                    (bounds_yaxis[1]-bounds_yaxis[0])+bounds_yaxis[0]
                if (project_minimum or project_mean) and GPBO.dim > 2:
                    inner_grid = []
                    for d in range(GPBO.dim):
                        if d == dim_xaxis:
                            inner_grid.append([x_grid[n, dim_xaxis]])
                        elif d == dim_yaxis:
                            inner_grid.append([x_grid[n, dim_yaxis]])
                        else:
                            inner_grid.append(np.linspace(GPBO.history[epoch]['bounds'][d, 0],
                                                          GPBO.history[epoch]['bounds'][d, 1],
                                                          grid_points_each_dim))

                    inner_grid = np.meshgrid(*inner_grid)
                    inner_grid = np.array(list(list(x.flat)
                                          for x in inner_grid), np.float32).T
                    inner_grid_n = torch.tensor(GPBO.normalize(
                        inner_grid, GPBO.history[epoch]['bounds']), dtype=torch.float32)

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

        if (not (project_minimum or project_mean)) or GPBO.dim == 2:
            if fixed_values_for_each_dim is not None:
                for dim, val in fixed_values_for_each_dim.items():
                    x_grid[:, dim] = val

            x_grid_n = torch.tensor(GPBO.normalize(
                x_grid, GPBO.history[epoch - 1]['bounds']), dtype=torch.float32)
            y_posterior = gp(x_grid_n)
            y_grid = -y_posterior.mean.detach().numpy()

        x = GPBO.history[epoch]['x']

        if i_query is None:
            x1 = GPBO.history[epoch]['x1']
            if len(x1) > 0:
                x1 = np.vstack(x1)
            else:
                x1 = None

            X_pending = []
            for xp_ in GPBO.history[epoch]['X_pending']:
                if xp_ is not None:
                    X_pending.append(xp_)
            if len(X_pending) > 0:
                X_pending = np.vstack(X_pending)
            else:
                X_pending = None
        else:
            if i_query >= len(GPBO.history[epoch]['x1']):
                raise ValueError(
                    'i_query exceed number of query in the epoch '+str(epoch))
            x1 = GPBO.history[epoch]['x1'][i_query]
            X_pending = GPBO.history[epoch]['X_pending'][i_query]
        
        if self.posterior_ax:
            axes = [self.posterior_ax]
            
        for ax in axes:
            data = ax.tricontourf(
                x_grid[:, dim_xaxis], x_grid[:, dim_yaxis], y_grid, levels=64, cmap="viridis")
            tag = ""
            ax.scatter(x[:, dim_xaxis], x[:, dim_yaxis],
                    c="b", alpha=0.7, label="training data")
            if x1 is not None:
                ax.scatter(x1[:, dim_xaxis], x1[:, dim_yaxis],
                        c="r", alpha=0.7, label="candidate")
            if X_pending is not None:
                X_pending = np.atleast_2d(X_pending)
                ax.scatter(X_pending[:, dim_xaxis], X_pending[:,
                        dim_yaxis], s=50, c="r", marker='x', label="pending")
                
            self._handle_colorbar(ax, data, y_grid)
    def plot_obj_history(self, GPBO, plot_best_only=False, axes: list = []) -> None:
        if not self.obj_history_ax and not axes:
            raise ValueError("No axes provided and `self.acquisition_ax` is `None`.")
        
        y_best_hist = [np.min(GPBO.history[-1]['y'][:i+1])
                       for i in range(len(GPBO.history[-1]['y']))]
            
        if self.obj_history_ax:
            axes = [self.obj_history_ax]
        
        for ax in axes:
            ax.plot(y_best_hist, color='C0', label='Cumulative Minimum Loss')
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
                ax1.plot(GPBO.history[-1]['y'], color='C1', label='Loss')


class NavigationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent):
        self.toolitems = (('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Pan', 'Left button pans, Right button zooms\nx/y fixes axis, CTRL fixes aspect', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
        )
        super().__init__(canvas, parent)
        self.update()


class PlotsCanvas(BaseCanvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MAIN FIGURE
        self.figure = Figure()
        
        # Separate into 3 x 2 grid
        gs = self.figure.add_gridspec(3, 2)
        
        # Create axes on grid
        self.acquisition_ax = self.figure.add_subplot(gs[:2, 0])
        self.posterior_ax = self.figure.add_subplot(gs[:2, 1])
        self.obj_history_ax = self.figure.add_subplot(gs[2, :])
        
        self.hide_axes()
    def format(self, acq_xylabels: Tuple[str, str], post_xylabels: Tuple[str, str], legends=True) -> None:
        """
        Applies the canvas format.
        
        For labels, element 0 is the x-label and element 1 is the y-label.

        Args:
            acq_xy_labels (Tuple[str, str]): The acquisition axes XY-labels.
            post_xy_labels (Tuple[str, str]): The posterior mean axes XY-labels.

        Raises:
            ValueError: If any axes do not exist.
        """
        if not self.acquisition_ax or not self.posterior_ax or not self.obj_history_ax:
            raise ValueError("One or more of the required axes are missing.")
        
        # INDICES
        X = 0
        Y = 1
        
        # TEXT SIZE
        TITLE_SIZE = 12
        LABEL_SIZE = 11
        TICK_SIZE = 10
        
        # TEXT
        ACQ_TITLE = "Acquisition"
        POST_TITLE = "Posterior Mean"
        OBJ_TITLE = "Objective History"
        OBJ_X_LABEL = "Evaluation Budget"
        OBJ_Y_LABEL = "Objective Minimum"
        OBJ_TWIN_LABEL = "Loss"
        
        twinx = self._get_obj_twinx()
        
        self.acquisition_ax.set_title(ACQ_TITLE, fontsize=TITLE_SIZE)
        self.acquisition_ax.set_xlabel(acq_xylabels[X], fontsize=LABEL_SIZE)
        self.acquisition_ax.set_ylabel(acq_xylabels[Y], fontsize=LABEL_SIZE)
        self.acquisition_ax.tick_params(axis="both", labelsize=TICK_SIZE)
        
        self.posterior_ax.set_title(POST_TITLE, fontsize=TITLE_SIZE)
        self.posterior_ax.set_xlabel(post_xylabels[X], fontsize=LABEL_SIZE)
        self.posterior_ax.set_ylabel(post_xylabels[Y], fontsize=LABEL_SIZE)
        self.posterior_ax.tick_params(axis="both", labelsize=TICK_SIZE)
                
        self.obj_history_ax.set_title(OBJ_TITLE, fontsize=TITLE_SIZE)
        self.obj_history_ax.set_xlabel(OBJ_X_LABEL, fontsize=LABEL_SIZE)
        self.obj_history_ax.set_ylabel(OBJ_Y_LABEL, fontsize=LABEL_SIZE)
        self.obj_history_ax.tick_params("both", labelsize=TICK_SIZE)
        if twinx:
            twinx.set_ylabel(OBJ_TWIN_LABEL, fontsize=LABEL_SIZE)
            twinx.tick_params(axis="both", labelsize=TICK_SIZE)

        if legends:
            self.posterior_ax.legend().set_draggable(True)
        
            if twinx:
                twinx_lines = twinx.get_lines()
                if twinx_lines:
                    obj_lineA = twinx_lines[0]
                    obj_lineB = self.obj_history_ax.get_lines()[0]
                    obj_lines = [obj_lineA, obj_lineB]
                    twinx.legend(obj_lines, [line.get_label() for line in obj_lines]).set_draggable(True)
    def reload(self) -> None:
        """Reloads the canvas."""
        for ax in self.get_axes():
            ax.relim()
            ax.autoscale()

        self.figure.tight_layout()
        self.figure.subplots_adjust(top=0.95)

        self.draw_idle()


class MainCanvas(BaseCanvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.axes = self.figure.subplots(3, 1)
        self.acquisition_ax = self.axes[0]
        self.posterior_ax = self.axes[1]
        self.obj_history_ax = self.axes[2]

        self.hide_axes()
    def format(self, acq_xylabels: Tuple[str, str], post_xylabels: Tuple[str, str], legends=False) -> None:
        """
        Applies the canvas format.
        
        For labels, element 0 is the x-label and element 1 is the y-label.

        Args:
            acq_xy_labels (Tuple[str, str]): The acquisition axes XY-labels.
            post_xy_labels (Tuple[str, str]): The posterior mean axes XY-labels.

        Raises:
            ValueError: If any axes do not exist.
        """
        if not self.acquisition_ax or not self.posterior_ax or not self.obj_history_ax:
            raise ValueError("One or more of the required axes are missing.")
        
        # INDICES
        X = 0
        Y = 1
        
        # TEXT SIZE
        TITLE_SIZE = 10
        LABEL_SIZE = 9
        TICK_SIZE = 8
        
        # TEXT
        ACQ_TITLE = "Acquisition"
        POST_TITLE = "Posterior Mean"
        OBJ_TITLE = "Objective History"
        OBJ_X_LABEL = "Evaluation Budget"
        OBJ_Y_LABEL = "Objective Minimum"
        OBJ_TWIN_LABEL = "Loss"
        
        twinx = self._get_obj_twinx()
        
        self.acquisition_ax.set_title(ACQ_TITLE, fontsize=TITLE_SIZE)
        self.acquisition_ax.set_xlabel(acq_xylabels[X], fontsize=LABEL_SIZE)
        self.acquisition_ax.set_ylabel(acq_xylabels[Y], fontsize=LABEL_SIZE)
        self.acquisition_ax.tick_params(axis="both", labelsize=TICK_SIZE)
        
        self.posterior_ax.set_title(POST_TITLE, fontsize=TITLE_SIZE)
        self.posterior_ax.set_xlabel(post_xylabels[X], fontsize=LABEL_SIZE)
        self.posterior_ax.set_ylabel(post_xylabels[Y], fontsize=LABEL_SIZE)
        self.posterior_ax.tick_params(axis="both", labelsize=TICK_SIZE)
                
        self.obj_history_ax.set_title(OBJ_TITLE, fontsize=TITLE_SIZE)
        self.obj_history_ax.set_xlabel(OBJ_X_LABEL, fontsize=LABEL_SIZE)
        self.obj_history_ax.set_ylabel(OBJ_Y_LABEL, fontsize=LABEL_SIZE)
        self.obj_history_ax.tick_params("both", labelsize=TICK_SIZE)
        if twinx:
            twinx.set_ylabel(OBJ_TWIN_LABEL, fontsize=LABEL_SIZE)
            twinx.tick_params(axis="both", labelsize=TICK_SIZE)

        if legends:
            self.posterior_ax.legend().set_draggable(True)
        
            if twinx:
                twinx_lines = twinx.get_lines()
                if twinx_lines:
                    obj_lineA = twinx_lines[0]
                    obj_lineB = self.obj_history_ax.get_lines()[0]
                    obj_lines = [obj_lineA, obj_lineB]
                    twinx.legend(obj_lines, [line.get_label() for line in obj_lines]).set_draggable(True)
    def reload(self) -> None:
        """Reloads the canvas."""
        for ax in self.get_axes():
            ax.relim()
            ax.autoscale()

        self.figure.tight_layout()

        self.draw_idle()
