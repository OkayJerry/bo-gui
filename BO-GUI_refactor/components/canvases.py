
#############
# Functions #
#############

def plot(self, main_window, projection=True):
    canvas = main_window.plots_page.canvas
    preview_canvas = main_window.plots_page.preview_canvas
    gpbo = main_window.interactiveGPBO
    
    # Clear the canvases
    preview_canvas.clear()
    canvas.clear()
    
    # Plot projections and objective history
    if projection: # Takes a long time to calculate
        gpbo.plot_aquisition_2D_projection(
            epoch=gpbo.epoch, axes=gpbo.acq_axes)
        gpbo.plot_GPmean_2D_projection(
            epoch=gpbo.epoch, axes=gpbo.post_axes)
    gpbo.plot_obj_history(axes=gpbo.obj_axes)

    # Format and reload canvases
    preview_canvas.quickFormat(epoch=gpbo.epoch)
    canvas.quickFormat(epoch=gpbo.epoch)
    preview_canvas.reload()
    canvas.reload()

    # optional
    # tabs.setTabEnabled(1, True)
    # tabs.setTabEnabled(2, True)