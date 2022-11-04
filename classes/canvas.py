from matplotlib.backends.backend_qtagg import FigureCanvas


class PreviewCanvas(FigureCanvas):
    def __init__(self):
        super().__init__()

        self.axes = self.figure.subplots(3, 1)
        self.acquisition_ax = self.axes[0]
        self.posterior_ax = self.axes[1]
        self.obj_history_ax = self.axes[2]

        self.acquisition_ax.set_title('acquisition')
        self.posterior_ax.set_title('posterior mean')
        self.obj_history_ax.set_title('objective history')

        self.figure.tight_layout()
        self.draw_idle()

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def quickFormat(self, epoch=0, x_label='x1', y_label='x2'):
        import globals as glb

        self.acquisition_ax.set_xlabel(
            glb.main_window.acq_x_combobox.currentText())
        self.acquisition_ax.set_ylabel(
            glb.main_window.acq_y_combobox.currentText())
        self.posterior_ax.set_xlabel(
            glb.main_window.post_x_combobox.currentText())
        self.posterior_ax.set_ylabel(
            glb.main_window.post_y_combobox.currentText())
        self.obj_history_ax.set_xlabel("evaluation budget")
        self.obj_history_ax.set_ylabel("obj minimum")

        self.acquisition_ax.set_title(f"epoch {epoch}, aquisition")
        self.posterior_ax.set_title(f"epoch {epoch}, posterior mean")
        self.obj_history_ax.set_title(f"epoch {epoch}, objective history")
        self.reload()

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def reload(self):
        for ax in self.axes:
            ax.relim()
            ax.autoscale()

        obj_twin_ax = self.obj_history_ax.get_shared_x_axes(
        ).get_siblings(self.obj_history_ax)[0]
        obj_twin_ax.relim()
        obj_twin_ax.autoscale()

        self.figure.tight_layout()
        self.draw_idle()

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def clear(self):
        for ax in self.axes:
            ax.clear()

        self.obj_history_ax.get_shared_x_axes().get_siblings(
            self.obj_history_ax)[0].clear()


class Canvas(FigureCanvas):
    def __init__(self):
        super().__init__()

        gs = self.figure.add_gridspec(2, 2)
        self.acquisition_ax = self.figure.add_subplot(gs[0, 0])
        self.posterior_ax = self.figure.add_subplot(gs[0, 1])
        self.obj_history_ax = self.figure.add_subplot(gs[1, :])
        self.axes = [self.acquisition_ax,
                     self.posterior_ax, self.obj_history_ax]

        self.acquisition_ax.set_title('acquisition')
        self.posterior_ax.set_title('posterior mean')
        self.obj_history_ax.set_title('objective history')

        self.figure.tight_layout()
        self.draw_idle()

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def quickFormat(self, epoch=0):
        import globals as glb

        self.acquisition_ax.set_xlabel(
            glb.main_window.acq_x_combobox.currentText())
        self.acquisition_ax.set_ylabel(
            glb.main_window.acq_y_combobox.currentText())
        self.posterior_ax.set_xlabel(
            glb.main_window.post_x_combobox.currentText())
        self.posterior_ax.set_ylabel(
            glb.main_window.post_y_combobox.currentText())
        self.obj_history_ax.set_xlabel("evaluation budget")
        self.obj_history_ax.set_ylabel("obj minimum")

        self.posterior_ax.legend().set_draggable(True)

        obj_lineA = self.obj_history_ax.get_shared_x_axes(
        ).get_siblings(self.obj_history_ax)[0].get_lines()[0]
        obj_lineB = self.obj_history_ax.get_lines()[0]
        obj_lines = [obj_lineA, obj_lineB]
        self.obj_history_ax.legend(
            obj_lines, [line.get_label() for line in obj_lines]).set_draggable(True)

        self.acquisition_ax.set_title(f"epoch {epoch}, aquisition")
        self.posterior_ax.set_title(f"epoch {epoch}, posterior mean")
        self.obj_history_ax.set_title(f"epoch {epoch}, objective history")
        self.reload()

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def reload(self):
        for ax in self.axes:
            ax.relim()
            ax.autoscale()

        obj_twin_ax = self.obj_history_ax.get_shared_x_axes(
        ).get_siblings(self.obj_history_ax)[0]
        obj_twin_ax.relim()
        obj_twin_ax.autoscale()

        self.figure.tight_layout()
        self.draw_idle()

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def clear(self):
        for ax in self.axes:
            ax.clear()

        self.obj_history_ax.get_shared_x_axes().get_siblings(
            self.obj_history_ax)[0].clear()

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])
