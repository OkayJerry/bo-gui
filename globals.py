from classes.interactiveGPBO import InteractiveGPBO
import threading
import numpy as np

def calcDistance(pnt_A, pnt_B, weights):
    distances = []
    for i in range(len(pnt_A)):
        coord_A = pnt_A[i]
        coord_B = pnt_B[i]
        distances.append(abs(coord_A - coord_B) * weights[i])
    return max(distances)

def findBestPoint(prev_pnt, eval_pnts, weights):
    # calculating distances
    eval_pnt_distances = []
    for eval_pnt in eval_pnts:
            distance = calcDistance(prev_pnt, eval_pnt, weights)
            eval_pnt_distances.append(distance)
            
    pnt_index = eval_pnt_distances.index(min(eval_pnt_distances))
    return eval_pnts[pnt_index], pnt_index

def orderByDistance(prev_pnt, eval_pnts, weights):
    # print(f'Before:\n{eval_pnts}')
    eval_pnts = eval_pnts
    ordered_pnts = []
    while eval_pnts.size != 0:
        best_pnt, best_pnt_index = findBestPoint(prev_pnt, eval_pnts, weights)
        ordered_pnts.append(best_pnt)
        eval_pnts = np.delete(eval_pnts, best_pnt_index, 0)
    # print(f'After:\n{np.array(ordered_pnts)}')
    return np.array(ordered_pnts)

class GPBO(InteractiveGPBO):
    def __init__(self, x, y, bounds, batch_size=1, acquisition_func=None, acquisition_func_args=None,
                 acquisition_optimize_options={
                     "num_restarts": 20, "raw_samples": 20},
                 scipy_minimize_options=None, prior_mean_model=None, prior_mean_model_env=None,
                 L2reg=0.05, avoid_bounds_corners=True, path="./log/", tag=""):

        super().__init__(x, y, bounds, batch_size, acquisition_func, acquisition_func_args,
                         acquisition_optimize_options, scipy_minimize_options, prior_mean_model,
                         prior_mean_model_env, L2reg, avoid_bounds_corners, path, tag)

        eval_table = main_window.evaluation_point_groupbox.evaluation_point_table
        weight_table = main_window.weight_table
        preview_canvas = main_window.preview_canvas
        canvas = main_window.canvas

        self.acq_axes = [preview_canvas.acquisition_ax, canvas.acquisition_ax]
        self.post_axes = [preview_canvas.posterior_ax, canvas.posterior_ax]
        self.obj_axes = [preview_canvas.obj_history_ax, canvas.obj_history_ax]

        eval_x = self.step(batch_size=batch_size)
        eval_x = orderByDistance(self.x[-1], eval_x, weight_table.getWeights())
        eval_table.addEvaluationX(eval_x)

        thread = threading.Thread(target=self.plot_all)
        thread.start()

    def plot_all(self):
        preview_canvas = main_window.preview_canvas
        plot_button = main_window.plot_button
        canvas = main_window.canvas
        tabs = main_window.tabs

        # required
        preview_canvas.clear()
        canvas.clear()
        if plot_button.isChecked():
            self.plot_aquisition_2D_projection(
                epoch=self.epoch, axes=self.acq_axes)
            self.plot_GPmean_2D_projection(epoch=self.epoch, axes=self.post_axes)
        self.plot_obj_history(axes=self.obj_axes)

        preview_canvas.quickFormat(epoch=self.epoch)
        canvas.quickFormat(epoch=self.epoch)
        preview_canvas.reload()
        canvas.reload()

        # optional
        tabs.setTabEnabled(1, True)
        tabs.setTabEnabled(2, True)

    def run(self, x, y, batch_size, acq_args):
        eval_table = main_window.evaluation_point_groupbox.evaluation_point_table
        weight_table = main_window.weight_table
        epoch_label = main_window.epoch_label

        eval_x = self.step(x=x, y=y, batch_size=batch_size,
                           acquisition_func_args=acq_args)
        eval_x = orderByDistance(self.x[-1], eval_x, weight_table.getWeights())
        eval_table.addEvaluationX(eval_x)

        thread = threading.Thread(target=self.plot_all)
        thread.start()

        epoch_label.setText(str(self.epoch))


def init():
    from sys import argv
    from PyQt5.QtWidgets import QApplication
    from classes.windows import MainWindow
    import matplotlib as mpl

    global app
    global main_window
    global interactive_GPBO
    global acq_widgets
    global tables

    mpl.rc('font', size=8)
    mpl.rc('lines', markersize=3)
    acq_widgets = []
    tables = []
    interactive_GPBO = InteractiveGPBO(
        x=np.array([[]]), y=np.array([[]]), bounds=[])

    app = QApplication(argv)
    main_window = MainWindow()
