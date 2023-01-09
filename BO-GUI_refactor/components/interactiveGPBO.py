from classes.interactiveGPBO import InteractiveGPBO
import threading
import numpy as np

def calcDistance(pnt_A, pnt_B, weights):
    distances = []
    for i in range(len(pnt_A)):
        coord_A = pnt_A[i]
        coord_B = pnt_B[i]
        if weights:
            distances.append(abs(coord_A - coord_B) * weights[i])
        else:
            distances.append(abs(coord_A - coord_B))
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
    eval_pnts = eval_pnts
    ordered_pnts = []

    if len(eval_pnts[0]) != len(weights):
        weights = []

    while eval_pnts.size != 0:
        best_pnt, best_pnt_index = findBestPoint(prev_pnt, eval_pnts, weights)
        ordered_pnts.append(best_pnt)
        eval_pnts = np.delete(eval_pnts, best_pnt_index, 0)
    return np.array(ordered_pnts)

class GPBO(InteractiveGPBO):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
            
        if np.size(self.x) != 0:
            eval_table = main_window.evaluation_point_groupbox.evaluation_point_table

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