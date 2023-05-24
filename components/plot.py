from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple, List, Union

class BaseCanvas(FigureCanvasQTAgg):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.acquisition_ax: Axes = None
        self.posterior_ax: Axes = None
        self.obj_history_ax: Axes = None

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
            
    def format(self, acq_xy_labels: Tuple[str, str], post_xy_labels: Tuple[str, str], compact=False, legends=True):
        """
        Applies the canvas format.
        
        For labels, element 0 is the x-label and element 1 is the y-label.

        Args:
            acq_xy_labels (Tuple[str, str]): The acquisition axes XY-labels.
            post_xy_labels (Tuple[str, str]): The posterior mean axes XY-labels.
            compact (bool, optional): Reduces font size. Defaults to False.
            legends (_type_, optional): Whether to include legends. Defaults to true.

        Raises:
            ValueError: If any axes do not exist.
        """
        if not self.acquisition_ax or not self.posterior_ax or not self.obj_history_ax:
            raise ValueError("One or more of the required axes are missing.")
        
        # INDICES
        X = 0
        Y = 1
        
        # TEXT SIZE
        size_logic = lambda size: size - 2 if compact else size

        TITLE_SIZE = size_logic(12)
        LABEL_SIZE = size_logic(11)
        TICK_SIZE = size_logic(10)
        
        # TEXT
        ACQ_TITLE = "Acquisition"
        POST_TITLE = "Posterior Mean"
        OBJ_TITLE = "Objective History"
        OBJ_X_LABEL = "Evaluation Budget"
        OBJ_Y_LABEL = "Objective Minimum"
        OBJ_TWIN_LABEL = "Loss"
        
        twinx = self._get_obj_twinx()
        
        self.acquisition_ax.set_title(ACQ_TITLE, fontsize=TITLE_SIZE)
        self.acquisition_ax.set_xlabel(acq_xy_labels[X], fontsize=LABEL_SIZE)
        self.acquisition_ax.set_ylabel(acq_xy_labels[Y], fontsize=LABEL_SIZE)
        self.acquisition_ax.tick_params(axis="both", labelsize=TICK_SIZE)
        
        self.posterior_ax.set_title(POST_TITLE, fontsize=TITLE_SIZE)
        self.posterior_ax.set_xlabel(post_xy_labels[X], fontsize=LABEL_SIZE)
        self.posterior_ax.set_ylabel(post_xy_labels[Y], fontsize=LABEL_SIZE)
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
                
    def reload(self, compact=False) -> None:
        """
        Reloads the canvas.

        Args:
            compact (bool, optional): Adjusts the margin at the top of each subplot. Defaults to False.
        """
        for ax in self.get_axes():
            ax.relim()
            ax.autoscale()

        self.figure.tight_layout()
        if not compact:
            self.figure.subplots_adjust(top=0.95)

        self.draw_idle()

    def hide_axes(self) -> None:
        """
        Hides all axes.
        """
        for ax in self.get_axes():
            if ax.get_visible():
                ax.set_visible(False)

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
class MainCanvas(BaseCanvas):
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

class PreviewCanvas(BaseCanvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.axes = self.figure.subplots(3, 1)
        self.acquisition_ax = self.axes[0]
        self.posterior_ax = self.axes[1]
        self.obj_history_ax = self.axes[2]

        self.hide_axes()