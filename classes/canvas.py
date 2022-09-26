from matplotlib.backends.backend_qtagg import FigureCanvas
import matplotlib as mpl

class Canvas(FigureCanvas):
    def __init__(self):
        super().__init__()
        
        mpl.rc('font', size=8)
        self.axes = self.figure.subplots(2, 1)
        self.acquisition_ax = self.axes[0]
        self.posterior_ax = self.axes[1]
        
        self.acquisition_ax.set_title('acquisition')
        self.posterior_ax.set_title('posterior mean')

        self.figure.tight_layout()
        self.draw_idle()
        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3]) 

    def quickFormat(self, epoch=0):
        for ax in self.axes:
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            legend = ax.legend()
            legend.set_draggable(True)

        self.acquisition_ax.set_title("epoch "+str(epoch)+", aquisition")
        self.posterior_ax.set_title("epoch "+str(epoch)+", posterior mean")
        self.reload()
        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3]) 

    def reload(self):
        for ax in self.axes:
            ax.margins(1, 0)
            ax.relim()
            ax.autoscale()

        self.figure.tight_layout(h_pad=2)
        self.draw_idle()
        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3]) 

    def clear(self):
        for ax in self.axes:
            ax.clear()