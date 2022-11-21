from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QKeySequence
import numpy as np
import csv
import threading

class HorizontalHeaderContextMenu(QMenu):
    def __init__(self, index, parent=None):
        import globals as glb
        super().__init__(parent)

        rename_action = QAction('Rename', self)
        add_column_action = QAction('Add Column', self)
        insert_column_action = QAction('Insert Column', self)
        remove_column_action = QAction('Remove Column', self)
        
        rename_action.triggered.connect(lambda: self.rename(index))
        add_column_action.triggered.connect(self.addColumn)
        insert_column_action.triggered.connect(lambda: self.insertColumn(index))
        remove_column_action.triggered.connect(lambda: self.removeColumn(index))

        self.addAction(rename_action)
        
        if parent is not glb.main_window.iteration_x_table and parent is not glb.main_window.initial_y_table and parent is not glb.main_window.iteration_y_table:
            self.addSeparator()
            self.addAction(add_column_action)
            self.addAction(insert_column_action)
            self.addAction(remove_column_action)

        
    def rename(self, index):
        import globals as glb
        
        parent_table = self.parent()
        old_header = parent_table.horizontalHeaderItem(index).text()
        new_header, ok = QInputDialog.getText(self, 'Rename Header', f'Header {index + 1}:', QLineEdit.Normal, old_header)

        if ok:
            parent_table.horizontalHeaderItem(index).setText(new_header)
            if parent_table is glb.main_window.initial_x_table:
                glb.main_window.bounds_table.horizontalHeaderItem(index).setText(new_header)
                glb.main_window.weight_table.verticalHeaderItem(index).setText(new_header)
                
            elif parent_table is glb.main_window.iteration_x_table:
                eval_table = glb.main_window.evaluation_point_groupbox.evaluation_point_table
                acq_post_comboboxes = [glb.main_window.acq_x_combobox, glb.main_window.acq_y_combobox, glb.main_window.post_x_combobox, glb.main_window.post_y_combobox]
                
                eval_table.horizontalHeaderItem(index).setText(new_header)
                
                for combobox in acq_post_comboboxes:
                    combobox.blockSignals(True)
                    text_i = combobox.findText(old_header, Qt.MatchExactly)
                    combobox.setItemText(text_i, new_header)
                    combobox.blockSignals(False)
                    
                canvases = [glb.main_window.canvas, glb.main_window.preview_canvas]
                for canvas in canvases:
                    for ax in canvas.axes:
                        if ax.xaxis.get_label().get_text() == old_header:
                            ax.set_xlabel(new_header)
                        elif ax.yaxis.get_label().get_text() == old_header:
                            ax.set_ylabel(new_header)
                    canvas.reload()
                            
                acq_post_tables = [glb.main_window.acq_fixed_table, glb.main_window.post_fixed_table]
                for table in acq_post_tables:
                    table.verticalHeaderItem(index).setText(new_header)
                
                

    def insertColumn(self, index):
        parent_table = self.parent()
        parent_table.insertColumn(index)
        self.handleTableSynchronization(index)

    def removeColumn(self, index):
        parent_table = self.parent()
        parent_table.removeColumn(index)
        self.handleTableSynchronization(index, column_removal=True)
        
    def addColumn(self):
        parent_table = self.parent()
        parent_table.insertColumn(parent_table.columnCount())
        self.handleTableSynchronization(parent_table.columnCount() - 1)

    def handleTableSynchronization(self, index, column_removal=False):
        import globals as glb
        
        parent_table = self.parent()
        if not column_removal:
            if parent_table == glb.main_window.initial_x_table:
                glb.main_window.bounds_table.insertColumn(index)
            elif parent_table == glb.main_window.bounds_table:
                glb.main_window.initial_x_table.insertColumn(index)
                
            glb.main_window.weight_table.insertRow(index)
        else:
            if parent_table == glb.main_window.initial_x_table:
                glb.main_window.bounds_table.removeColumn(index)
            elif parent_table == glb.main_window.bounds_table:
                glb.main_window.initial_x_table.removeColumn(index)

            glb.main_window.weight_table.removeRow(index)

            glb.main_window.bounds_table.horizontalHeader().resizeSections(QHeaderView.Stretch)
            glb.main_window.initial_x_table.horizontalHeader().resizeSections(QHeaderView.Stretch)


class VerticalHeaderContextMenu(QMenu):
    def __init__(self, index, parent=None):
        super().__init__(parent)

        remove_row_action = QAction('Remove Row', self)
        
        remove_row_action.triggered.connect(lambda: self.removeRow(index))

        self.addAction(remove_row_action)


    def removeRow(self, index):
        parent_table = self.parent()
        if index != parent_table.rowCount() - 1:
            parent_table.removeRow(index)
            self.handleTableSynchronization(index, row_removal=True)
        else:
            QMessageBox.critical(parent_table, 'ERROR', 'Bottom row cannot be removed. If you wish to do so, clear it instead.', QMessageBox.Ok, QMessageBox.Ok)
        
    def handleTableSynchronization(self, index, row_removal=False):
        import globals as glb
        
        parent_table = self.parent()
        parent_table.blockSignals(True)
        if row_removal:
            if parent_table is glb.main_window.initial_x_table:
                glb.main_window.initial_y_table.removeRow(index)
            elif parent_table is glb.main_window.initial_y_table:
                glb.main_window.initial_x_table.removeRow(index)
            elif parent_table is glb.main_window.iteration_x_table:
                glb.main_window.iteration_y_table.removeRow(index)
            elif parent_table is glb.main_window.iteration_y_table:
                glb.main_window.iteration_x_table.removeRow(index)
        
class TableContextMenu(QMenu):
    def __init__(self, index, parent=None):
        super().__init__(parent=parent)

        clear_action = QAction('Clear', self)

        clear_action.triggered.connect(self.clearSelection)
        
        self.addAction(clear_action)
        
    def clearSelection(self):
        parent_table = self.parent()
        selected_items = parent_table.selectedItems()
        for item in selected_items:
            item.setText('')
            
        # parent_table.removeBlankRows()
            
            
class AcquisitionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.ucb_button = QRadioButton('Upper Confidence Bound (UCB)')
        self.ucb_button.setChecked(True)
        self.ucb_button.toggled.connect(self.handleBetaEnabling)
        self.ucb_spinbox = QDoubleSpinBox()
        self.kg_button = QRadioButton('Knowledge Gradient (KG)')
        
        layout.addWidget(self.ucb_button, 0, 0, 1, 6)
        layout.addWidget(QLabel('Beta:'), 1, 1, 1, 1, alignment=Qt.AlignRight)
        layout.addWidget(self.ucb_spinbox, 1, 2, 1, 1)
        layout.addWidget(self.kg_button, 2, 0, 1, 6)
        
        self.setLayout(layout)

    def handleBetaEnabling(self, checked):
        import globals as glb

        if checked:
            self.ucb_button.blockSignals(True)

            self.ucb_button.setChecked(True)
            self.ucb_spinbox.setEnabled(True)

            self.ucb_button.blockSignals(False)
        else:
            self.ucb_button.blockSignals(True)

            self.kg_button.setChecked(True)
            self.ucb_spinbox.setEnabled(False)

            self.ucb_button.blockSignals(False)
                
class IterateButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setText('Step Iteration')
        self.clicked.connect(self.onClick)

    def onClick(self):
        import globals as glb

        epoch_label = glb.main_window.epoch_label
        reg_coeff = glb.main_window.reg_coeff_spinbox.value()
        batch_size = glb.main_window.iteration_batch_spin_box.value()

        x, x_success = glb.main_window.iteration_x_table.getArray()
        y, y_success = glb.main_window.iteration_y_table.getArray()
        iteration_beta = glb.main_window.iteration_beta_spin_box

        if x_success and y_success:
            if int(epoch_label.text()) > 1:
                glb.interactive_GPBO.update_exploration(x, y)
            
            if iteration_beta.isEnabled():
                beta_value = np.float32(iteration_beta.value())
                acq_args = {'beta': beta_value}
            else:
                acq_args = {}

            # print(f'x: \n{x}\n',
            #     f'y: \n{y}\n',
            #     f'bounds: {bounds}\n',
            #     f'batch size: {batch_size}\n',
            #     f'acquisition function: {glb.interactive_GPBO.acquisition_func}\n',
            #     f'acquisition arguments: {acq_args}\n',
            #     f'Regularization Coefficient: {reg_coeff}')
            glb.main_window.progress_dialog.reset()
            glb.main_window.progress_dialog.setNumPlots(2)
            glb.interactive_GPBO.run(x, y, batch_size, acq_args)
        else:
             QMessageBox.critical(self, 'ERROR', 'Iteration array(s) incomplete.', QMessageBox.Ok)
        
        
class MenuBar(QMenuBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        main_window = parent
        
        # file menu
        new_action = QAction('&New', main_window)
        open_action = QAction('&Open...', main_window)
        save_action = QAction('&Save', main_window)
        save_as_action = QAction('&Save As...', main_window)
        exit_action = QAction('&Exit', main_window)
        
        new_action.setShortcut(QKeySequence.New)
        open_action.setShortcut(QKeySequence.Open)
        save_action.setShortcut(QKeySequence.Save)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        exit_action.setShortcut(QKeySequence.Quit)

        new_action.triggered.connect(self.newFile)
        open_action.triggered.connect(self.openFile)
        # save_action.triggered.connect(self.saveFile)
        # save_as_action.triggered.connect(self.saveFileAs)
        exit_action.triggered.connect(qApp.quit)

        file_menu = self.addMenu('File')
        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        file_menu.addAction(exit_action)

    def newFile(self):
        response = QMessageBox.warning(self, 'WARNING', f'Are you sure? Any unsaved progress will be lost.', QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        
        if response == QMessageBox.Ok:
            main_window = self.parent()
            main_window.returnToInitializiation()
        
    def openFile(self):
        from classes.tables import TableItem, DoubleTableItem
        import globals as glb
        
        main_window = self.parent()
        
        while True:

            filename = QFileDialog.getOpenFileName(self.parent(), 'Open File', filter="All Files (*.*);;CSV File (*.csv);;PICKLE File (*.pickle)")[0]  # previously tuple
            if filename:
                if ".csv" not in filename and ".pickle" not in filename:
                    warning = QMessageBox()
                    warning.setIcon(QMessageBox.Critical)
                    warning.setText("Didn't select a .csv or .pickle file")
                    warning.setWindowTitle("ERROR")
                    warning.setStandardButtons(QMessageBox.Ok)

                    if warning.exec() == QMessageBox.Ok:
                        warning.close()
                        continue
            break

        if not filename:
            return
        elif ".csv" in filename:
            reader = csv.reader(open(filename), delimiter=",")
            
            # necessary because reader object is unscriptable
            initial_data = []
            for row in reader:
                initial_data.append(row)

            # formatting
            data = {'xheader': [],
                                'yheader': 'y0',
                                'xdata': [],
                                'ydata': [],
                                'bounds': {'min': [],
                                        'max': []}}
            for element in initial_data[0][:-1]:
                data['xheader'].append(element)
            data['yheader'] = initial_data[0][-1]
            for row in initial_data[1:-3]:
                data['xdata'].append(row[:-1])
                data['ydata'].append(row[-1])
            data['bounds']['min'] = initial_data[-2][:-1]
            data['bounds']['max'] = initial_data[-1][:-1]
        elif ".pickle" in filename:
            # x, y, bounds, batch_size = glb.interactive_GPBO.load_from_log(filename)
            for element in glb.interactive_GPBO.load_from_log(filename):
                print(element)
                print('*****')        
        
        # resetting initialization windows
        main_window.returnToInitializiation()
        
        # setting initialization tables
        initial_x_table = main_window.initial_x_table
        initial_y_table = main_window.initial_y_table
        boundry_table = main_window.bounds_table
        weight_table = main_window.weight_table

        initial_x_table.setColumnCount(len(data['xheader']))
        initial_x_table.setRowCount(len(data['xdata']))
        for i, header in enumerate(data['xheader']):
            initial_x_table.setHorizontalHeaderItem(i, TableItem(header))
        for i, row in enumerate(data['xdata']):
            for j, val in enumerate(row):
                initial_x_table.setItem(i, j, DoubleTableItem(val))
 
        initial_y_table.setRowCount(len(data['xdata']))
        initial_y_table.horizontalHeaderItem(0).setText(data['yheader'])
        for i, val in enumerate(data['ydata']):
            initial_y_table.setItem(i, 0, DoubleTableItem(val))
                    
        boundry_table.setColumnCount(len(data['bounds']['min']))
        for i, header in enumerate(data['xheader']):
            boundry_table.setHorizontalHeaderItem(i, TableItem(header))
        for i, val in enumerate(data['bounds']['min']):
            boundry_table.setItem(0, i, DoubleTableItem(val))
        for i, val in enumerate(data['bounds']['max']):
            boundry_table.setItem(1, i, DoubleTableItem(val))
            
        weight_table.setRowCount(len(data['xheader']))
        for i, header in enumerate(data['xheader']):
            weight_table.setItem(i, 0, DoubleTableItem(1))
            weight_table.setVerticalHeaderItem(i, TableItem(header))
            
class InitializeButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setText('Initialize and Iterate Once')
        self.clicked.connect(self.onClick)

    def onClick(self):
        import globals as glb
        from globals import GPBO

        
        if glb.main_window.tabs.isTabEnabled(1):
            reply = QMessageBox.question(self,
                                         'Important',
                                         "Initializing will remove any iterations. Are you sure?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
            else:
                glb.main_window.preview_canvas.clear()

        bounds, bounds_success = glb.main_window.bounds_table.getBounds()
        epoch_label = glb.main_window.epoch_label
        reg_coeff = glb.main_window.reg_coeff_spinbox.value()

        x0, x0_success = glb.main_window.initial_x_table.getArray()
        y0, y0_success = glb.main_window.initial_y_table.getArray()
        
        if x0_success and y0_success and bounds_success:
            ucb_box = glb.main_window.initial_acq_widget.ucb_button
            
            if ucb_box.isChecked():
                beta_value = np.float32(glb.main_window.initial_acq_widget.ucb_spinbox.value())
                acq_func = 'UCB'
                acq_args = {'beta': beta_value}
            else:
                acq_func = 'KG'
                acq_args = {}           

            coords_not_in_bounds = []
            for i, row in enumerate(x0):
                for j, val in enumerate(row):
                    if val < bounds[j][0] or val > bounds[j][1]:
                        coords_not_in_bounds.append(f'({j + 1}, {i + 1})')
                        
            if coords_not_in_bounds:
                message = 'Values at '
                if len(coords_not_in_bounds) > 1:
                    for coord in coords_not_in_bounds:
                        if coord != coords_not_in_bounds[-1] and len(coords_not_in_bounds) > 2:
                            message += coord + ', '
                        elif coord != coords_not_in_bounds[-1]:
                            message += coord + ' '
                        else:
                            message += 'and ' + coord
                    message += ' need to be exclusively within bounds.'
                else:
                        message = f'Value at {coords_not_in_bounds[0]} needs to be exclusively within bounds.'
                
                QMessageBox.critical(self, 'ERROR', message, QMessageBox.Ok)
                return

            # print(f'x0: {x0}\n',
            #       f'y0: {y0}\n',
            #       f'bounds: {bounds}\n',
            #       f'batch size: {1}\n',
            #       f'acquisition function: {acq_func}\n',
            #       f'acquisition arguments: {acq_args}\n',
            #       f'Regularization Coefficient: {reg_coeff}')

            progress_dialog = glb.main_window.progress_dialog
            progress_dialog.setNumPlots(2)
            glb.interactive_GPBO = GPBO(x0, y0, bounds=bounds, acquisition_func=acq_func,
                                        acquisition_func_args=acq_args, L2reg=reg_coeff)

            epoch_label.setText('1')
            glb.main_window.transferDataFromInitialization()
            progress_dialog.reset()
            glb.interactive_GPBO.updateProgressBar.connect(progress_dialog.handle)
        else:
            QMessageBox.critical(self, 'ERROR', 'Initial array(s) incomplete.', QMessageBox.Ok)
        
class EvaluationPointGroupBox(QGroupBox):
        def __init__(self, parent=None):
            from classes.tables import EvaluationPointTable

            super().__init__('Evaluation Point', parent=parent)
            
            self.evaluation_point_table = EvaluationPointTable(1)
            self.add_points_button = QPushButton('Add All Points')
            self.add_point_button = QPushButton('Add Selected Point(s)')

            self.add_points_button.clicked.connect(self.addPoints)
            self.add_point_button.clicked.connect(self.addPoint)
            
            layout = QGridLayout()
            layout.addWidget(self.evaluation_point_table, 0, 0, 1, 2)
            layout.addWidget(self.add_points_button, 1, 0, 1, 1)
            layout.addWidget(self.add_point_button, 1, 1, 1, 1)

            self.setLayout(layout)

        def addPoints(self):
            import globals as glb
            
            iteration_x_table = glb.main_window.iteration_x_table
            iteration_y_table = glb.main_window.iteration_y_table
            
            og_points, og_success = iteration_x_table.getArray()
            n_points, n_success = self.evaluation_point_table.getArray(empty_last_row=False)

            
            if n_success and og_success:
                n_x = np.concatenate((og_points, n_points))
            elif not n_success:
                QMessageBox.critical(self, 'ERROR', 'No evaluation points available.', QMessageBox.Ok)
                return
            elif not og_success:
                QMessageBox.critical(self, 'ERROR', 'Iteration array(s) incomplete.', QMessageBox.Ok)
                return
            
            iteration_x_table.fillFromArray(n_x)

            # print(f'original points: \n{og_points}\n',
            #       f'new points: \n{n_points}\n',
            #       f'new x: \n{n_x}\n')
            
            self.evaluation_point_table.setRowCount(0)
            iteration_y_table.setRowCount(iteration_y_table.rowCount() + len(n_points))

            # for testing
            def rosenbrock(x_decision,x_env=None,noise=0.01):
                f = 0
                ndim = len(x_decision)
                for i in range(len(x_decision) -1):
                    f += (x_decision[i+1]-x_decision[i]**2)**2 + 0.01*(1-x_decision[i])**2
                if x_env is not None:
                    if type(x_env)==float:
                        f += (x_env-x_decision[-1]**2)**2 + 0.01*(1-x_decision[-1])**2
                        nenv = 1
                    else:
                        f += (x_env[0]-x_decision[-1]**2)**2 + 0.01*(1-x_decision[-1])**2
                        for i in range(1,len(x_env) -1):
                            f += (x_env[i+1]-x_env[i]**2)**2 + 0.01*(1-x_env[i])**2
                        nenv = len(x_env)
                else:
                    nenv = 0
                        
                return f/(ndim+nenv)  + np.random.randn()*noise
            
            og_y, _  = iteration_y_table.getArray()
            og_y = [row for row in og_y if row != []]
            og_y = np.array(og_y)

            array = []
            for point in n_x[len(og_y):]:
                array.append([rosenbrock(point)])
            array = np.array(array)
            n_y = np.concatenate((og_y, array))
            iteration_y_table.fillFromArray(n_y)

        def addPoint(self):
            import globals as glb
            from classes.tables import DoubleTableItem
            
            iteration_x_table = glb.main_window.iteration_x_table
            iteration_y_table = glb.main_window.iteration_y_table

            selection = self.evaluation_point_table.selectionModel().selectedRows()
            
            if len(selection) == 0:
                QMessageBox.critical(self, 'ERROR', 'No evaluation points selected.', QMessageBox.Ok)
                return
                
            pnts = [] # for testing
            for model_index in selection:
                pnt = [] # for testing
                for i in range(self.evaluation_point_table.columnCount()):
                    val = self.evaluation_point_table.item(model_index.row(), i).text()
                    iteration_x_table.setItem(iteration_x_table.rowCount() - 1, i, DoubleTableItem(val))
                    pnt.append(np.float32(val)) # for testing
                pnts.append(pnt)
                
            for model_index in selection:
                self.evaluation_point_table.removeRow(model_index.row())

            iteration_y_table.setRowCount(iteration_y_table.rowCount() + 1)
            
            # for testing
            def rosenbrock(x_decision,x_env=None,noise=0.01):
                f = 0
                ndim = len(x_decision)
                for i in range(len(x_decision) -1):
                    f += (x_decision[i+1]-x_decision[i]**2)**2 + 0.01*(1-x_decision[i])**2
                if x_env is not None:
                    if type(x_env)==float:
                        f += (x_env-x_decision[-1]**2)**2 + 0.01*(1-x_decision[-1])**2
                        nenv = 1
                    else:
                        f += (x_env[0]-x_decision[-1]**2)**2 + 0.01*(1-x_decision[-1])**2
                        for i in range(1,len(x_env) -1):
                            f += (x_env[i+1]-x_env[i]**2)**2 + 0.01*(1-x_env[i])**2
                        nenv = len(x_env)
                else:
                    nenv = 0
                        
                return f/(ndim+nenv)  + np.random.randn()*noise
            
            og_y, _  = iteration_y_table.getArray()
            og_y = [row for row in og_y if row != []]
            og_y = np.array(og_y)
            array = []
            for pnt in pnts:
                array.append([rosenbrock(pnt)])
            array = np.array(array)
            n_y = np.concatenate((og_y, array))
            iteration_y_table.fillFromArray(n_y)

class AcqPostButton(QRadioButton):
    def __init__(self, text: str):
        super().__init__(text)
        
        self.pressed.connect(self.onClick)
        
    def onClick(self):
        import globals as glb
        
        preview_canvas = glb.main_window.preview_canvas
        canvas = glb.main_window.canvas
            
        self.setChecked(True)
        
        acq_axes = [preview_canvas.acquisition_ax, canvas.acquisition_ax]
        post_axes = [preview_canvas.posterior_ax, canvas.posterior_ax]
        epoch = int(glb.main_window.epoch_label.text())
        
        if self is glb.main_window.acq_min_button:
            x_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_x_combobox.currentText())
            y_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_y_combobox.currentText())
            glb.main_window.acq_fixed_table.setEnabled(False)
            glb.main_window.acq_fix_update_button.setEnabled(False)
            for ax in acq_axes:
                ax.clear()
            glb.main_window.progress_dialog.setNumPlots(1)
            glb.interactive_GPBO.plot_aquisition_2D_projection(dim_xaxis=x_axis, dim_yaxis=y_axis, epoch=epoch, axes=acq_axes, project_minimum=True)
            
        elif self is glb.main_window.acq_mean_button:
            x_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_x_combobox.currentText())
            y_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_y_combobox.currentText())
            glb.main_window.acq_fixed_table.setEnabled(False)
            glb.main_window.acq_fix_update_button.setEnabled(False)
            for ax in acq_axes:
                ax.clear()
            glb.main_window.progress_dialog.setNumPlots(1)
            glb.interactive_GPBO.plot_aquisition_2D_projection(dim_xaxis=x_axis, dim_yaxis=y_axis, epoch=epoch, axes=acq_axes, project_mean=True, project_minimum=False)

        elif self is glb.main_window.acq_fix_button:
            x_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_x_combobox.currentText())
            y_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_y_combobox.currentText())
            glb.main_window.acq_fixed_table.setEnabled(True)
            glb.main_window.acq_fix_update_button.setEnabled(True)
            
            for ax in acq_axes:
                ax.clear()
            d = glb.main_window.acq_fixed_table.getFixedDimValues()
            glb.main_window.progress_dialog.setNumPlots(1)
            glb.interactive_GPBO.plot_aquisition_2D_projection(axes=acq_axes, project_minimum=False, project_mean=False, fixed_values_for_each_dim=d)
        
        elif self is glb.main_window.post_min_button:
            x_axis = glb.main_window.post_fixed_table.getAxisIndex(glb.main_window.post_x_combobox.currentText())
            y_axis = glb.main_window.post_fixed_table.getAxisIndex(glb.main_window.post_y_combobox.currentText())
            glb.main_window.post_fixed_table.setEnabled(False)
            glb.main_window.post_fix_update_button.setEnabled(False)
            for ax in post_axes:
                ax.clear()
            glb.main_window.progress_dialog.setNumPlots(1)
            glb.interactive_GPBO.plot_GPmean_2D_projection(dim_xaxis=x_axis, dim_yaxis=y_axis, epoch=epoch, axes=post_axes, project_minimum=True)
            
        elif self is glb.main_window.post_mean_button:
            x_axis = glb.main_window.post_fixed_table.getAxisIndex(glb.main_window.post_x_combobox.currentText())
            y_axis = glb.main_window.post_fixed_table.getAxisIndex(glb.main_window.post_y_combobox.currentText())
            glb.main_window.post_fixed_table.setEnabled(False)
            glb.main_window.post_fix_update_button.setEnabled(False)
            for ax in post_axes:
                ax.clear()
            glb.main_window.progress_dialog.setNumPlots(1)
            glb.interactive_GPBO.plot_GPmean_2D_projection(dim_xaxis=x_axis, dim_yaxis=y_axis, epoch=epoch, axes=post_axes, project_mean=True, project_minimum=False)
            
        elif self is glb.main_window.post_fix_button:
            x_axis = glb.main_window.post_fixed_table.getAxisIndex(glb.main_window.post_x_combobox.currentText())
            y_axis = glb.main_window.post_fixed_table.getAxisIndex(glb.main_window.post_y_combobox.currentText())
            glb.main_window.post_fixed_table.setEnabled(True)
            glb.main_window.post_fix_update_button.setEnabled(True)
            
            for ax in post_axes:
                ax.clear()
            d = glb.main_window.acq_fixed_table.getFixedDimValues()
            glb.main_window.progress_dialog.setNumPlots(1)
            glb.interactive_GPBO.plot_GPmean_2D_projection(axes=post_axes, project_minimum=False, project_mean=False, fixed_values_for_each_dim=d)
        
        preview_canvas.quickFormat(epoch=epoch)
        canvas.quickFormat(epoch=epoch)
        preview_canvas.reload()
        canvas.reload()

        
class AxisComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        self.currentTextChanged.connect(self.onTextChange)
        self.prev_text = None
        
    def onTextChange(self, current_text):
        import globals as glb
        
        self.blockSignals(True)
        
        if self.prev_text == None:
            if self is glb.main_window.acq_x_combobox or self is glb.main_window.post_x_combobox:
                self.prev_text = self.itemText(0)
            else:
                self.prev_text = self.itemText(1)
            
        if self is glb.main_window.acq_x_combobox:
            paired_combobox = glb.main_window.acq_y_combobox
            # for canvas in [glb.main_window.preview_canvas, glb.main_window.canvas]:
            #     canvas.acquisition_ax.set_xlabel(current_text)
                
        elif self is glb.main_window.acq_y_combobox:
            paired_combobox = glb.main_window.acq_x_combobox
            # for canvas in [glb.main_window.preview_canvas, glb.main_window.canvas]:
            #     canvas.acquisition_ax.set_ylabel(current_text)
                
        elif self is glb.main_window.post_x_combobox:
            paired_combobox = glb.main_window.post_y_combobox
            # for canvas in [glb.main_window.preview_canvas, glb.main_window.canvas]:
            #     canvas.posterior_ax.set_xlabel(current_text)
                
        else:
            paired_combobox = glb.main_window.post_x_combobox
            # for canvas in [glb.main_window.preview_canvas, glb.main_window.canvas]:
            #     canvas.posterior_ax.set_ylabel(current_text)
            
        if current_text == paired_combobox.currentText():
            self.setCurrentText(self.prev_text)
            warning = QMessageBox()
            warning.setIcon(QMessageBox.Critical)
            warning.setText("Cannot select two identical axis.")
            warning.setWindowTitle("ERROR")
            warning.setStandardButtons(QMessageBox.Ok)

            if warning.exec() == QMessageBox.Ok:
                warning.close()
                self.blockSignals(False)
                return

        if self is glb.main_window.acq_x_combobox or self is glb.main_window.acq_y_combobox:
            table = glb.main_window.acq_fixed_table
        else:
            table = glb.main_window.post_fixed_table
            
        for row_n in range(table.rowCount()):
            if current_text == table.verticalHeaderItem(row_n).text():
                table.hideRow(row_n)
                break
            
        if current_text != paired_combobox.currentText():
            table.returnAxis(self.prev_text)
            self.prev_text = current_text

        acq_axes = [glb.main_window.preview_canvas.acquisition_ax, glb.main_window.canvas.acquisition_ax]
        post_axes = [glb.main_window.preview_canvas.posterior_ax, glb.main_window.canvas.posterior_ax]
        epoch = int(glb.main_window.epoch_label.text())
        
        if self is glb.main_window.acq_x_combobox or self is glb.main_window.acq_y_combobox:
            x_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_x_combobox.currentText())
            y_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_y_combobox.currentText())
            for i, button in enumerate([glb.main_window.acq_min_button, glb.main_window.acq_mean_button, glb.main_window.acq_fix_button]):
                if button.isChecked():
                    if i == 0:
                        for ax in acq_axes:
                            ax.clear()
                        glb.main_window.progress_dialog.setNumPlots(1)
                        glb.interactive_GPBO.plot_aquisition_2D_projection(dim_xaxis=x_axis, dim_yaxis=y_axis, epoch=epoch, axes=acq_axes, project_minimum=True)
                    elif i == 1:
                        for ax in acq_axes:
                            ax.clear()
                        glb.main_window.progress_dialog.setNumPlots(1)
                        glb.interactive_GPBO.plot_aquisition_2D_projection(dim_xaxis=x_axis, dim_yaxis=y_axis, epoch=epoch, axes=acq_axes, project_mean=True, project_minimum=False)
                    else:
                        pass # TBD
        else:
            for i, button in enumerate([glb.main_window.post_min_button, glb.main_window.post_mean_button, glb.main_window.post_fix_button]):
                x_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_x_combobox.currentText())
                y_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_y_combobox.currentText())
                if button.isChecked():
                    if i == 0:
                        for ax in post_axes:
                            ax.clear()
                        glb.main_window.progress_dialog.setNumPlots(1)
                        glb.interactive_GPBO.plot_GPmean_2D_projection(dim_xaxis=x_axis, dim_yaxis=y_axis, epoch=epoch, axes=post_axes, project_minimum=True)
                    elif i == 1:
                        for ax in post_axes:
                            ax.clear()
                        glb.main_window.progress_dialog.setNumPlots(1)
                        glb.interactive_GPBO.plot_GPmean_2D_projection(dim_xaxis=x_axis, dim_yaxis=y_axis, epoch=epoch, axes=post_axes, project_mean=True, project_minimum=False)
                    else:
                        pass # TBD
                    
        glb.main_window.canvas.quickFormat(epoch=int(glb.main_window.epoch_label.text()))
        glb.main_window.preview_canvas.quickFormat(epoch=int(glb.main_window.epoch_label.text()))

        self.blockSignals(False)
        
class AcqPostFixUpdateButton(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent=parent)
        
        self.pressed.connect(self.onPress)
        self.setEnabled(False)
    
    def onPress(self):
        import globals as glb
        
        acq_axes = [glb.main_window.preview_canvas.acquisition_ax, glb.main_window.canvas.acquisition_ax]
        post_axes = [glb.main_window.preview_canvas.posterior_ax, glb.main_window.canvas.posterior_ax]
        epoch = int(glb.main_window.epoch_label.text())
        
        if self is glb.main_window.acq_fix_update_button:
            d = glb.main_window.acq_fixed_table.getFixedDimValues()
            x_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_x_combobox.currentText())
            y_axis = glb.main_window.acq_fixed_table.getAxisIndex(glb.main_window.acq_y_combobox.currentText())
            for ax in acq_axes:
                ax.clear()
            glb.main_window.progress_dialog.setNumPlots(1)
            glb.interactive_GPBO.plot_aquisition_2D_projection(dim_xaxis=x_axis, dim_yaxis=y_axis, axes=acq_axes, project_minimum=False, project_mean=False, fixed_values_for_each_dim=d)
        else:
            d = glb.main_window.post_fixed_table.getFixedDimValues()
            x_axis = glb.main_window.post_fixed_table.getAxisIndex(glb.main_window.post_x_combobox.currentText())
            y_axis = glb.main_window.post_fixed_table.getAxisIndex(glb.main_window.post_y_combobox.currentText())
            for ax in post_axes:
                ax.clear()
            glb.main_window.progress_dialog.setNumPlots(1)
            glb.interactive_GPBO.plot_GPmean_2D_iprojection(dim_xaxis=x_axis, dim_yaxis=y_axis, axes=post_axes, project_minimum=False, project_mean=False, fixed_values_for_each_dim=d)
            
        glb.main_window.preview_canvas.quickFormat(epoch=epoch)
        glb.main_window.canvas.quickFormat(epoch=epoch)
        glb.main_window.preview_canvas.reload()
        glb.main_window.canvas.reload()

class ProgressDialog(QProgressDialog):
    def __init__(self, parent=None):
        super().__init__('Loading...', 'Cancel', 1, 1020, parent)
        
        self.cancel() # necessary because QProgressDialog opens up automatically following construction/initialization
        self.setWindowTitle('Optimization')
        self.setWindowModality(Qt.WindowModal)
        self.setCancelButton(None)

    def handle(self, value: int, num_plots=2):
        if not self.isVisible():
            self.show()
        
        self.setValue(self.value() + value)

    def setNumPlots(self, num_plots=2):
        if num_plots == 2 and self.maximum() != 1020:
            self.setMaximum(1020)
        elif num_plots == 1 and self.maximum() != 510:
            self.setMaximum(510)