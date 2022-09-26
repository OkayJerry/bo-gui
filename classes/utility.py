from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QKeySequence
import numpy as np
import csv
from copy import deepcopy

class HeaderContextMenu(QMenu):
    def __init__(self, index, parent=None):
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
        else:
            if parent_table == glb.main_window.initial_x_table:
                glb.main_window.bounds_table.removeColumn(index)
            elif parent_table == glb.main_window.bounds_table:
                glb.main_window.initial_x_table.removeColumn(index)
            
            
            
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
        # from botorch.acquisition import qUpperConfidenceBound, qKnowledgeGradient
        
        bounds = glb.main_window.bounds_table.getBounds()
        epoch_label = glb.main_window.epoch_label
        reg_coeff = glb.main_window.reg_coeff_spinbox.value()
        batch_size = glb.main_window.iteration_batch_spin_box.value()

        x = glb.main_window.iteration_x_table.getArray()
        y = glb.main_window.iteration_y_table.getArray()
        iteration_beta = glb.main_window.iteration_beta_spin_box
        
        if iteration_beta.isEnabled():
            beta_value = np.float32(iteration_beta.value())
            acq_args = {'beta': beta_value}
        else:
            acq_args = {}

        print(f'x: \n{x}\n',
              f'y: \n{y}\n',
              f'bounds: {bounds}\n',
              f'batch size: {batch_size}\n',
              f'acquisition function: {glb.interactive_GPBO.acquisition_func}\n',
              f'acquisition arguments: {acq_args}\n',
              f'Regularization Coefficient: {reg_coeff}')
        
        eval_x = glb.interactive_GPBO.step(x=x, y=y, batch_size=batch_size, acquisition_func_args=acq_args)
        glb.main_window.evaluation_point_groupbox.evaluation_point_table.addEvaluationX(eval_x)

        canvas = glb.main_window.canvas
        canvas.clear()
        glb.interactive_GPBO.plot_aquisition_2D_projection(epoch=int(epoch_label.text()) + 1, ax=canvas.acquisition_ax)
        glb.interactive_GPBO.plot_GPmean_2D_projection(epoch=int(epoch_label.text()) + 1, ax=canvas.posterior_ax)
        canvas.quickFormat(epoch=int(epoch_label.text()) + 1)
        canvas.reload()

        epoch_label.setText(str(int(epoch_label.text()) + 1))
        
        
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

        # new_action.triggered.connect(self.newFile)
        open_action.triggered.connect(self.openFile)
        # save_action.triggered.connect(self.saveFile)
        # save_as_action.triggered.connect(self.saveFileAs)
        exit_action.triggered.connect(qApp.quit)

        file_menu = self.addMenu('File')
        file_menu.addAction(open_action)
        file_menu.addAction(exit_action)

        
    def openFile(self):
        import globals as glb
        
        main_window = self.parent()
        
        while True:
            filename = QFileDialog.getOpenFileName(self.parent(), 'Open File', filter="CSV File (*.csv)")[0]  # previously tuple
            if filename:
                if ".csv" not in filename:
                    warning = QMessageBox()
                    warning.setIcon(QMessageBox.Critical)
                    warning.setText("Didn't select a .csv file")
                    warning.setWindowTitle("ERROR")
                    warning.setStandardButtons(QMessageBox.Ok)

                    if warning.exec() == QMessageBox.Ok:
                        warning.close()
                        continue
            break

        if not filename:
            return
        
        reader = csv.reader(open(filename), delimiter=",")
        
        # necessary because reader object is unscriptable
        data = []
        for row in reader:
            data.append(row)

        reformatted_data = {'xheader': [],
                            'yheader': 'y0',
                            'xdata': [],
                            'ydata': [],
                            'bounds': {'min': [],
                                    'max': []}}
        
        for element in data[0][:-1]:
            reformatted_data['xheader'].append(element)
            
        reformatted_data['yheader'] = data[0][-1]
            
        for row in data[1:-3]:
            reformatted_data['xdata'].append(row[:-1])
            reformatted_data['ydata'].append(row[-1])

        reformatted_data['bounds']['min'] = data[-2][:-1]
        reformatted_data['bounds']['max'] = data[-1][:-1]
        
        # setting initialization tables
        initial_x_table = main_window.initial_x_table
        initial_y_table = main_window.initial_y_table
        boundry_table = main_window.bounds_table

        # if len(reformatted_data['xheader']) != initial_x_table.columnCount():
        initial_x_table.setColumnCount(len(reformatted_data['xheader']))
            
        # if len(reformatted_data['xdata']) != initial_x_table.rowCount():
        initial_x_table.setRowCount(len(reformatted_data['xdata']))
        
        for i, header in enumerate(reformatted_data['xheader']):
            item = QTableWidgetItem()
            item.setText(header)
            item.setTextAlignment(Qt.AlignCenter)
            initial_x_table.setHorizontalHeaderItem(i, item)
        
        for i, row in enumerate(reformatted_data['xdata']):
            for j, column in enumerate(row):
                item = QTableWidgetItem()
                item.setText(column)
                item.setTextAlignment(Qt.AlignCenter)
                initial_x_table.setItem(i, j, item)
 
        # if len(reformatted_data['ydata']) != initial_y_table.rowCount():
        initial_y_table.setRowCount(len(reformatted_data['xdata']))
            
        initial_y_table.horizontalHeaderItem(0).setText(reformatted_data['yheader'])
            
        for i, val in enumerate(reformatted_data['ydata']):
            item = QTableWidgetItem()
            item.setText(val)
            item.setTextAlignment(Qt.AlignCenter)
            initial_y_table.setItem(i, 0, item)
                    
        # if len(reformatted_data['bounds']['min']) != boundry_table.columnCount():
        boundry_table.setColumnCount(len(reformatted_data['bounds']['min']))
            
        for i, header in enumerate(reformatted_data['xheader']):
            item = QTableWidgetItem()
            item.setText(header)
            boundry_table.setHorizontalHeaderItem(i, item)
                
        for i, column in enumerate(reformatted_data['bounds']['min']):
            item = QTableWidgetItem()
            item.setText(column)
            item.setTextAlignment(Qt.AlignCenter)
            boundry_table.setItem(0, i, item)
            
        for i, column in enumerate(reformatted_data['bounds']['max']):
            item = QTableWidgetItem()
            item.setText(column)
            item.setTextAlignment(Qt.AlignCenter)
            boundry_table.setItem(1, i, item)

class GPBOWorker(QObject):
    finished = pyqtSignal()
    
    def runInit(self, 
                x, y,
                bounds,
                batch_size = 1,
                acquisition_func = None,
                acquisition_func_args = None,
                acquisition_optimize_options = {"num_restarts":20, "raw_samples":20},#, "nonlinear_inequality_constraints":[]},
                scipy_minimize_options=None,
                prior_mean_model=None,
                prior_mean_model_env=None,
                L2reg = 0.05,
                avoid_bounds_corners = True,
                path="./log/",
                tag=""):
        
        from classes.interactiveGPBO import InteractiveGPBO
        import globals as glb
        
        glb.interactive_GPBO = InteractiveGPBO(x, y,
                                               bounds=bounds,
                                               batch_size=batch_size,
                                               acquisition_func=acquisition_func,
                                               acquisition_func_args=acquisition_func_args,
                                               acquisition_optimize_options=acquisition_optimize_options,
                                               scipy_minimize_options=scipy_minimize_options,
                                               prior_mean_model=prior_mean_model,
                                               prior_mean_model_env=prior_mean_model_env,
                                               L2reg=L2reg,
                                               avoid_bounds_corners=avoid_bounds_corners,
                                               path=path,
                                               tag=tag)
        
        glb.interactive_GPBO.step(batch_size=1)
        
        self.finished.emit()
        
    # def runStep(self)

        
        
class InitializeButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setText('Initialize')
        self.clicked.connect(self.onClick)

    def handleThread(self,
                     canvas,
                     epoch,
                     x, y,
                     bounds,
                     batch_size = 1,
                     acquisition_func = None,
                     acquisition_func_args = None,
                     acquisition_optimize_options = {"num_restarts":20, "raw_samples":20},#, "nonlinear_inequality_constraints":[]},
                     scipy_minimize_options=None,
                     prior_mean_model=None,
                     prior_mean_model_env=None,
                     L2reg = 0.05,
                     avoid_bounds_corners = True,
                     path="./log/",
                     tag=""):
        
        self.thread = QThread()
        self.worker = GPBOWorker()
        

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(lambda: self.worker.runInit(x, y,
                                                                bounds=bounds,
                                                                batch_size=batch_size,
                                                                acquisition_func=acquisition_func,
                                                                acquisition_func_args=acquisition_func_args,
                                                                acquisition_optimize_options=acquisition_optimize_options,
                                                                scipy_minimize_options=scipy_minimize_options,
                                                                prior_mean_model=prior_mean_model,
                                                                prior_mean_model_env=prior_mean_model_env,
                                                                L2reg=L2reg,
                                                                avoid_bounds_corners=avoid_bounds_corners,
                                                                path=path,
                                                                tag=tag))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.worker.deleteLater)
        
        import globals as glb
        self.thread.finished.connect(lambda: glb.interactive_GPBO.plot_aquisition_2D_projection(epoch=epoch, ax=canvas.acquisition_ax))
        self.thread.finished.connect(lambda: glb.interactive_GPBO.plot_GPmean_2D_projection(epoch=epoch, ax=canvas.posterior_ax))
        self.thread.finished.connect(lambda: canvas.quickFormat(epoch=epoch))
        self.thread.finished.connect(lambda: glb.main_window.tabs.setTabEnabled(1, True))
        
        self.thread.finished.connect(lambda: self.setEnabled(True))

        self.thread.start()
        self.setEnabled(False)



        
    def onClick(self):
        import globals as glb
        
        if glb.main_window.tabs.isTabEnabled(1):
            reply = QMessageBox.question(self,
                                         'Important',
                                         "Initializing will remove any iterations. Are you sure?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
            else:
                glb.main_window.canvas.clear()

        bounds = glb.main_window.bounds_table.getBounds()
        epoch_label = glb.main_window.epoch_label
        reg_coeff = glb.main_window.reg_coeff_spinbox.value()
        canvas = glb.main_window.canvas

        x0 = glb.main_window.initial_x_table.getArray()
        y0 = glb.main_window.initial_y_table.getArray()
        ucb_box = glb.main_window.initial_acq_widget.ucb_button
        
        if ucb_box.isChecked():
            beta_value = np.float32(glb.main_window.initial_acq_widget.ucb_spinbox.value())
            acq_func = 'UCB'
            acq_args = {'beta': beta_value}
        else:
            acq_func = 'KG'
            acq_args = {}           

        print(f'x0: {x0}\n',
              f'y0: {y0}\n',
              f'bounds: {bounds}\n',
              f'batch size: {1}\n',
              f'acquisition function: {acq_func}\n',
              f'acquisition arguments: {acq_args}\n',
              f'Regularization Coefficient: {reg_coeff}')
        
        epoch = 1
        self.handleThread(canvas, epoch, x0, y0, bounds, batch_size=1, acquisition_func=acq_func, acquisition_func_args=acq_args, L2reg=reg_coeff)

        epoch_label.setText(str(epoch))
        glb.main_window.transferDataFromInitializationToIteration()
        
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
            
            og_points = iteration_x_table.getArray()
            n_points = self.evaluation_point_table.getArray(empty_row=False)

            n_x = np.concatenate((og_points, n_points))
            iteration_x_table.fillFromArray(n_x)

            print(f'original points: \n{og_points}\n',
                  f'new points: \n{n_points}\n',
                  f'new x: \n{n_x}\n')
            
            self.evaluation_point_table.setRowCount(0)
            iteration_y_table.setRowCount(iteration_y_table.rowCount() + len(n_points))


        def addPoint(self):
            import globals as glb
            
            iteration_x_table = glb.main_window.iteration_x_table
            iteration_y_table = glb.main_window.iteration_y_table

            selection = self.evaluation_point_table.selectionModel().selectedRows()
            
            for model_index in selection:
                for i in range(self.evaluation_point_table.columnCount()):
                    val = self.evaluation_point_table.item(model_index.row(), i).text()
                    item = QTableWidgetItem()
                    item.setText(val)
                    item.setTextAlignment(Qt.AlignCenter)
                    iteration_x_table.setItem(iteration_x_table.rowCount() - 1, i, item)
                
            for model_index in selection:
                self.evaluation_point_table.removeRow(model_index.row())

            iteration_y_table.setRowCount(iteration_y_table.rowCount() + 1)