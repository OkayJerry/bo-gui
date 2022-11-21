from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

class NavigationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent):
        # it is important that this is set prior to initialization
        self.toolitems = (('Home', 'Reset original view', 'home', 'home'),
                          ('Back', 'Back to previous view', 'back', 'back'),
                          ('Forward', 'Forward to next view', 'forward', 'forward'),
                          (None, None, None, None),
                          ('Pan', 'Left button pans, Right button zooms\nx/y fixes axis, CTRL fixes aspect', 'move', 'pan'),
                          ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'),
                          ('Subplots', 'Configure plot', 'subplots', 'configure_subplots'),
                          # ('Color', 'Select line color', "qt4_editor_options", 'select_line_color'), # how to do a custom button: (Name, Tooltip, Unknown, Linked Method Name)
                          (None, None, None, None),
                          ('Save', 'Save the figure', 'filesave', 'save_figure'))
        
        super().__init__(canvas, parent)
        self.update()

    #     self.line_combo = QComboBox()
    #     select_button = QPushButton()

    #     select_button.setText('Select Current Line')
    #     select_button.clicked.connect(self.choose_color)

    #     self.select_window = QWidget()
    #     self.select_window.setWindowTitle('Line Select')
    #     self.select_window.setMinimumWidth(200)
    #     self.select_window.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
    #     self.select_window.setLayout(QVBoxLayout())
    #     self.select_window.layout().addWidget(self.line_combo)
    #     self.select_window.layout().addWidget(select_button)

    # def select_line_color(self):
    #     canvas = self.parent().parent().parent().canvas
        
    #     self.line_combo.clear()
    #     for param in canvas.lines.keys():
    #         self.line_combo.addItem(param)

    #     if self.line_combo.count() == 0:
    #         warning = QMessageBox()
    #         warning.setIcon(QMessageBox.Critical)
    #         warning.setText("No lines visible")
    #         warning.setWindowTitle("ERROR")
    #         warning.setStandardButtons(QMessageBox.Ok)

    #         if warning.exec() == QMessageBox.Ok:
    #             warning.close()
    #             return

    #     self.select_window.setWindowState(self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive) # restoring to maximized/normal state
    #     self.select_window.activateWindow()
    #     self.select_window.show()

    # def choose_color(self):
    #     canvas = self.parent().parent().parent().canvas
        
    #     self.select_window.close()
    #     for ln in canvas.lines.values():
    #         if self.line_combo.currentText() == ln.get_label():
    #             color = QColorDialog.getColor(title=ln.get_label())
    #             if color.isValid():
    #                 canvas.custom_colors[ln.get_label()] = color.name()
    #                 ln.set_color(color.name())
    #                 canvas.refresh()
    #             break

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
        
        # setting labels
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

        # setting titles
        self.acquisition_ax.set_title(f"epoch {epoch}, aquisition")
        self.posterior_ax.set_title(f"epoch {epoch}, posterior mean")
        self.obj_history_ax.set_title(f"epoch {epoch}, objective history")
        
        # finalizing
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
        import globals as glb
        
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

        # import numpy as np
        # # adding colorbar
        # for ax in [self.acquisition_ax, self.posterior_ax]:
        #     divider = make_axes_locatable(ax1)
        #     cax = divider.append_axes('right', size='5%', pad=0.05)
        #     self.figure.colorbar(im1, ax=ax, orientation='vertical')

            
        # self.figure.colorbar(orientation='vertical')
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
        self.figure.subplots_adjust(top=0.99)
        self.draw_idle()

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])

    def clear(self):
        import globals as glb
        
        for ax in self.axes:
            ax.clear()

        self.obj_history_ax.get_shared_x_axes().get_siblings(
            self.obj_history_ax)[0].clear()
        
       

        # import inspect
        # print(inspect.stack()[0][3])
        # print(inspect.stack()[1][3])
