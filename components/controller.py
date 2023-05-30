import json
import pickle
import threading
from typing import List, Union, NoReturn
import traceback

import numpy as np
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow

from components.model import interactiveGPBO, prior_mean_model_wrapper
from components.plot import *
from components.utils import *
from components.view import *

DTYPE = np.float64

class Controller(QMainWindow):
    startup = pyqtSignal()
    updatedGPBO = pyqtSignal(int)
    plotStarted = pyqtSignal()
    plotFinished = pyqtSignal(int)
    queryStarted = pyqtSignal()
    queryFinished = pyqtSignal(list)
    
    def __init__(self, app, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        TITLE: str = "BO-GUI"
        ICON: str = "images/frib.png"
        WIDTH: int = 640
        HEIGHT: int = 480
                
        self.setWindowTitle(TITLE)
        self.setWindowIcon(QIcon(ICON))
        self.resize(WIDTH, HEIGHT)
                
        self.app = app
        self.preferences = QSettings("Facility for Rare Isotope Beams", TITLE, parent=self)
        self.menu_bar = MenuBar(parent=self)
        self.status_bar = QStatusBar()
        self.main = MainView()
        self.plots = PlotsView()
        
        self.setMenuBar(self.menu_bar)
        self.setStatusBar(self.status_bar)
        self.setCentralWidget(CentralView(parent=self))

        self.set_initial_preferences()
        self.set_initial_states()
        self.set_connections()
    def set_initial_preferences(self) -> None:
        DEFAULT = {"App Font Size": 9,
                   "Log Data Path": os.getcwd() + "\log",
                   "Prior Mean Model Path": os.getcwd() + "\prior"}
        
        # use defaults if no setting history
        if not self.preferences.contains("App Font Size"):
            self.preferences.setValue("App Font Size", DEFAULT["App Font Size"])
        if not self.preferences.contains("Log Data Path"):
            self.preferences.setValue("Log Data Path", DEFAULT["Log Data Path"])
        if not self.preferences.contains("Prior Mean Model Path"):
            self.preferences.setValue("Prior Mean Model Path", DEFAULT["Prior Mean Model Path"])
            
        self.app.setStyleSheet("QWidget {font-size: " + str(self.preferences.value("App Font Size")) + "pt;}")
    def set_initial_states(self) -> None:
        EPOCH: int = 0
        NUM_DIMENSIONS: int = 4
        BATCH_SIZE: int = 1
        BETA: float = 2.0
        ROW_COUNT = 200
        
        assert EPOCH > -1
        assert NUM_DIMENSIONS > 1
        assert BATCH_SIZE > -1
        
        DECISION_HEADER: List[str] = [f"x[{i}]" for i in range(NUM_DIMENSIONS)]
        OBJECTIVE_HEADER: List[str] = ["y"]
        FIXED_HEADER: List[str] = ["Fix Value"]
        BOUNDRY_HEADER: List[str] = ["min", "max"]
        
        assert len(DECISION_HEADER) > 0
        assert len(OBJECTIVE_HEADER) == 1
        assert len(FIXED_HEADER) == 1
        assert len(BOUNDRY_HEADER) == 2
        
        self.GPBO = None
        self.prior_mean_model = None

        for canvas in [self.main.canvas, self.plots.canvas]:
            for cbar in canvas.colorbars.values():
                cbar.remove()
            canvas.colorbars.clear()
            
        self.centralWidget().tabs.setCurrentIndex(0)  # `0` = "Main"
        self.centralWidget().tabs.setTabEnabled(1, False)  # `1` = "Plots"

        # Main Page
        self.main.blockWidgetSignals(True)
        
        self.main.epoch_label.setText(str(EPOCH))
        self.main.plot_button.setChecked(False)
        self.main.ucb_button.setChecked(True)
        self.main.query_candidates_button.setEnabled(False)
        self.main.batch_spin_box.setValue(BATCH_SIZE)
        self.main.batch_spin_box.setRange(1, 100)
        self.main.batch_spin_box.setEnabled(False)
        self.main.beta_spinbox.setValue(BETA)
        self.main.beta_spinbox.setEnabled(True)
        
        self.main.candidate_pnt_table.clear()
        self.main.candidate_pnt_table.setRowCount(0)
        self.main.candidate_pnt_table.setColumnCount(NUM_DIMENSIONS)
        self.main.candidate_pnt_table.setHorizontalHeaderLabels(DECISION_HEADER)
        self.main.candidate_pnt_table.setEnabled(False)
        
        self.main.pending_pnt_table.clear()
        self.main.pending_pnt_table.setRowCount(0)
        self.main.pending_pnt_table.setColumnCount(NUM_DIMENSIONS)
        self.main.pending_pnt_table.setHorizontalHeaderLabels(DECISION_HEADER)
        self.main.pending_pnt_table.setEnabled(False)
        
        self.main.x_table.clear()
        self.main.x_table.setColumnCount(NUM_DIMENSIONS)
        self.main.x_table.setRowCount(ROW_COUNT)
        self.main.x_table.setHorizontalHeaderLabels(DECISION_HEADER)
        self.main.x_table.enableColumnChanges(True)
            
        self.main.y_table.clear()
        self.main.y_table.setColumnCount(1)
        self.main.y_table.setRowCount(ROW_COUNT)
        self.main.y_table.setHorizontalHeaderLabels(OBJECTIVE_HEADER)

        self.main.boundry_table.clear()
        self.main.boundry_table.setColumnCount(NUM_DIMENSIONS)
        self.main.boundry_table.setRowCount(2)
        self.main.boundry_table.setHorizontalHeaderLabels(DECISION_HEADER)
        self.main.boundry_table.setVerticalHeaderLabels(BOUNDRY_HEADER)
        self.main.boundry_table.enableColumnChanges(True)
        
        self.main.canvas.hide_axes()
        self.main.canvas.draw_idle()

        self.main.blockWidgetSignals(False)
        
        # Plots Page
        self.plots.blockWidgetSignals(True)
        
        self.plots.acq_min_button.setChecked(True)
        self.plots.post_min_button.setChecked(True)
        
        self.plots.blockWidgetSignals(True)
    def set_connections(self) -> None:
        
        # LOGIC
        def movePoints(src: BaseTable, dest: BaseTable, selected: bool = False) -> None:
            """
            Moves the points from one to another. Copies the points if the 
            source table is of type `CandidateTable`, takes the points if the 
            source table is of type `PendingTable`.
            """
            src.blockSignals(True)
            dest.blockSignals(True)

            if isinstance(src, PendingTable):
                items = src.take_items(selected)
                src.remove_empty_rows(True)
            elif isinstance(src, CandidateTable):
                items = src.copy_items(selected)

            dest.append_items(items)
            
            # Handle y table automatic row addition
            if isinstance(dest, XTable):
                self.main.y_table.blockSignals(True)
                self.main.y_table.setRowCount(dest.rowCount())
                self.main.y_table.blockSignals(False)
            
            # Allow the signal to operate again
            dest.blockSignals(False)
            src.blockSignals(False)
        def onGotObjectiveFunction(func_str: str) -> None:
            r"""
            `function` is a string-version of the code within the function.
            
            An example function might be...
            ```
            def objective_function(X):
                return X
            ```
            ...which as a string is "def objective_function(X):\n\treturn X".
            
            Presents a pop-up if an exception occurs, otherwise it applies the
            function to the corresponding objective value of each completed 
            decision parameter.
            """
            self.main.y_table.blockSignals(True)

            glbs = {}
            lcls = {}

            try:
                exec(func_str, glbs, lcls)
                custom_function = list(lcls.values())[0]

                # Get decision parameters (account for extra row)
                x = self.main.x_table.to_list()

                # Verify that all elements are complete
                for i, pnt in enumerate(x):
                    # Ignore incomplete points
                    if any(val is None for val in pnt):
                        continue
                    
                    # Ignore any filled table items
                    item = self.main.y_table.item(i, 0)
                    if item is None or item.text() == "":
                        # Apply custom function on point
                        obj_val = custom_function(np.array(pnt, dtype=DTYPE))
                        # Set into Y table
                        self.main.y_table.setItem(i, 0, CenteredTableItem(str(obj_val)))
            except Exception as exc:
                print(exc)
                critical = QMessageBox(self)
                critical.setWindowTitle('ERROR')
                critical.setText(f'Function crashed: {type(exc)}')
                critical.setDetailedText(str(exc))
                critical.setIcon(QMessageBox.Critical)
                critical.show()

                if critical.exec() == QMessageBox.Ok:
                    critical.close()
                    self.main.obj_func_win.setWindowState(self.main.obj_func_win.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)  # restoring to normal state
                    self.main.obj_func_win.activateWindow()
                    self.main.obj_func_win.show()
                
            # Allow the signal to operate again
            self.main.y_table.blockSignals(False)
        def onPlotsComboBoxChange(src: MemoryComboBox, other: MemoryComboBox) -> None:
            """
            Whenever the source combobox changes, the other combobox must be 
            assigned a new index if it is the same as the source. In this edge-
            case, I chose to have them swap values.
            """
            src.blockSignals(True)
            other.blockSignals(True)
                        
            prev = src.previousIndex()

            if src.currentIndex() == other.currentIndex():
                other.setPreviousIndex(prev)
                other.setCurrentIndex(prev)

            src.setPreviousIndex(src.currentIndex())

            if src is self.plots.acq_x_combobox or src is self.plots.acq_y_combobox:
                fixed_value_table = self.plots.acq_fixed_table
            elif src is self.plots.post_x_combobox or src is self.plots.post_y_combobox:
                fixed_value_table = self.plots.post_fixed_table
                
            textA = src.currentText()
            textB = other.currentText()
            labels = fixed_value_table.get_vertical_labels()
            for i, label in enumerate(labels):
                if label == textA or label == textB:
                    fixed_value_table.hideRow(i)
                elif fixed_value_table.isRowHidden(i):
                    fixed_value_table.showRow(i)
                

            other.blockSignals(False)
            src.blockSignals(False)
        def onPlotsEpochChange(epoch: Union[str, int]) -> None:
            """
            Adds/removes query numbers from the query combobox depending upon the
            current selected epoch.
            """
            if not self.GPBO:
                return
            
            print("Plots Epoch Change...")
            epoch = int(epoch)
            candidate_pnts = self.GPBO.history[epoch]["x1"]

            print("\tQuery ComboBox")
            for i in range(1, self.plots.query_combobox.count()):
                print(f"\t\tRemove Query {i}")
                self.plots.query_combobox.removeItem(1)

            if len(candidate_pnts) > 1:
                for i in range(len(candidate_pnts)):
                    print(f"\t\tAdd Query {i}")
                    self.plots.query_combobox.addItem(f"{i + 1}")
        def onNewFileRequest() -> None:
            """
            Restores the program to its initial state.
            """
            self.set_initial_states()
        def onOpenFileRequest() -> None:
            """
            Opens a file dialog until it is closed or the correct file extension
            is selected. With the correct file extension, it will initialize the
            GPBO and enable the necessary widgets for the first epoch.
            """
            while True:
                filename = QFileDialog.getOpenFileName(self, 'Open File', filter="All Files (*.*);;JSON File (*.json)")[0]  # previously tuple
                if filename:
                    _, extension = os.path.splitext(filename)
                    if extension != ".json":
                        warning = QMessageBox()
                        warning.setIcon(QMessageBox.Critical)
                        warning.setText("Didn't select a *.json file")
                        warning.setWindowTitle("ERROR")
                        warning.setStandardButtons(QMessageBox.Ok)

                        if warning.exec() == QMessageBox.Ok:
                            warning.close()
                            continue
                break

            if not filename:
                return
            
            self.set_initial_states()

            self.GPBO = interactiveGPBO(self, load_log_fname=filename)
                        
            # Fill arrays using X and Y data
            self.main.x_table.fill(self.GPBO.x)
            self.main.y_table.fill(self.GPBO.y)
            
            # Rotate array to fit bounds table and fill
            bounds = np.rot90(self.GPBO.bounds, k=-1)
            self.main.boundry_table.fill(bounds)

            # Get candidates from GPBO.history[-1]['x1'] to put into candidate points
            
            # Get pending from GPBO.history[-1]['X_pending'] to put into pending points
        
            
            # Reset initialization table labels
            x_labels = [f'x[{i}]' for i in range(self.GPBO.dim)]
            self.main.x_table.setHorizontalHeaderLabels(x_labels)
            self.main.y_table.setHorizontalHeaderLabels(['y'])
            
            # Adjust query tables
            for table in [self.main.candidate_pnt_table, self.main.pending_pnt_table, self.main.boundry_table]:
                table.setColumnCount(len(self.GPBO.bounds))
                table.setHorizontalHeaderLabels(x_labels)
            
            # Match batch size and epoch
            self.main.batch_spin_box.setValue(self.GPBO.batch_size)
            self.main.epoch_label.setText(str(len(self.GPBO.history)-1))
            
            # Move tab to 'Main'
            self.centralWidget().tabs.setCurrentIndex(0)
            
            # Use acquisition type to enable/disable the 'beta' parameter
            if self.GPBO.acquisition_func == qUpperConfidenceBound:
                self.main.ucb_button.setChecked(True)
                self.main.beta_spinbox.setEnabled(True)
                if 'beta' in self.GPBO.acquisition_func_args.keys():
                    self.main.beta_spinbox.setValue(self.GPBO.acquisition_func_args['beta'])
            elif self.GPBO.acquisition_func == qKnowledgeGradient:
                self.main.kg_button.setChecked(True)
                self.main.beta_spinbox.setEnabled(False)

            # Ensure tables are fitted correctly
            for tbl in [self.main.x_table, self.main.y_table, self.main.boundry_table, self.main.candidate_pnt_table, self.main.pending_pnt_table]:
                tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

            self.updatedGPBO.emit(self.GPBO.epoch())
        def onPlotRefreshRequest() -> None:
            """
            Reloads all canvases.
            """
            self.main.canvas.reload()
            self.plots.canvas.reload()
        def onSaveFileAsRequest() -> None:
            """
            Presents a filedialog that allows the user to save the GPBO and saves it.
            """
            GPBO = self.GPBO
            if not GPBO:
                return

            filename = QFileDialog.getSaveFileName(self, "Save As", filter="All Files (*.*);;JSON File (*.json);;PICKLE File (*.pickle)")[0]  # previously tuple
            path, tag = os.path.split(filename)
            path += '/' if not path.endswith('/') else ''
            
            # From GPBO.write_log()...
            data = []
            for hist in GPBO.history:
                hist_ = {}
                for key, val in hist.items():
                    if type(val) == np.ndarray:
                        val = val.tolist()
                    if key == 'gp':
                        hist_[key] = None
                    elif key == 'acquisition':
                        hist_[key] = val
                        if val is not None and type(val) is not str:
                            _ = str(type(val))
                            hist_[key] = _[_.rfind('.')+1:-2]
                        else:
                            hist_[key] = 'UCB'
                    else:
                        hist_[key] = val
                data.append(hist_)
            
            if tag.endswith(".pickle"):
                with open(path + tag, "wb") as file:
                    pickle.dump(data, file)
            elif tag.endswith(".json"):
                with open(path + tag, "w") as file:
                    json.dump(data, file)
        def onOpenPreferencesRequest() -> None:
            self.menu_bar.pref_win.show()
        def app_font_size_change(size: int) -> None:
            self.preferences.setValue("App Font Size", size)
            self.app.setStyleSheet("QWidget {font-size: " + str(size) + "pt;}")
        def verify_path(line_edit: MemoryLineEdit) -> None:
            path = line_edit.text()
            if os.path.exists(path):
                self.preferences.setValue("Log Data Path", path)
                line_edit.previous = path
                return
            QMessageBox.critical(self.menu_bar.pref_win, "ERROR", "Invalid file path.", QMessageBox.Ok)
            line_edit.setText(line_edit.previous)
        def onTableDimensionsChanged(table: QTableWidget, command_code: int, index: int) -> None:
            """
            Depending upon the command used, handles the response of widgets
            which are affected.
            """
            ROW_INSERTED, ROW_REMOVED = 0, 1
            COLUMN_INSERTED, COLUMN_REMOVED = 2, 3
            TABLES = [self.main.x_table, self.main.y_table, self.main.boundry_table, self.main.candidate_pnt_table, self.main.pending_pnt_table]

            if command_code == ROW_INSERTED:
                if isinstance(table, XTable):
                    self.main.y_table.insertRow(index)
                elif isinstance(table, YTable):
                    self.main.x_table.insertRow(index)
            elif command_code == ROW_REMOVED:
                if isinstance(table, XTable):
                    self.main.y_table.removeRow(index)
                elif isinstance(table, YTable):
                    self.main.x_table.removeRow(index)
            elif command_code == COLUMN_INSERTED:
                # Automatically format default header items
                for i, label in enumerate(table.get_horizontal_labels()):
                    if not label or label[:2] + label[-1] == "x[]":
                        item = QTableWidgetItem(f"x[{i}]")
                        table.setHorizontalHeaderItem(i, item)

                labels = table.get_horizontal_labels()
                for tbl in TABLES:
                    if tbl is not table and tbl is not self.main.y_table:
                        tbl.insertColumn(index)
                        tbl.setHorizontalHeaderLabels(labels)
                        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            elif command_code == COLUMN_REMOVED:
                # Automatically format default header items
                for i, label in enumerate(table.get_horizontal_labels()):
                    if not label or label[:2] + label[-1] == "x[]":
                        item = QTableWidgetItem(f"x[{i}]")
                        table.setHorizontalHeaderItem(i, item)
                        
                labels = table.get_horizontal_labels()
                for tbl in TABLES:
                    if tbl is not table and tbl is not self.main.y_table:
                        tbl.removeColumn(index)
                        tbl.setHorizontalHeaderLabels(labels)
                        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        def onGotPriorFunction(func_str: str) -> None:
            r"""
            `function` is a string-version of the code within the function.
            
            An example function might be...
            ```
            def objective_function(X):
                return X
            ```
            ...which as a string is "def objective_function(X):\n\treturn X".
            """
            glbs = {}
            lcls = {}

            try:
                exec(func_str, glbs, lcls)
                custom_function = list(lcls.values())[0]
                
                test_dim = 2, self.main.x_table.rowCount()  # batch size x decision parameter dimension
                test_pnt = np.random.random(test_dim)
                test_result = custom_function(test_pnt)
                
                if test_result is None:
                    warning = QMessageBox(self)
                    warning.setWindowTitle("WARNING")
                    warning.setIcon(QMessageBox.Warning)
                    warning.setText("Return value of `None` cannot be utilized. Press 'OK' if you would like to make changes, otherwise the prior mean model will be discarded.")
                    warning.setStandardButtons(QMessageBox.Ok)
                    
                    if warning.exec() == QMessageBox.Ok:
                        self.menu_bar.prior_win.setWindowState(self.menu_bar.prior_win.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)  # restoring to normal state
                        self.menu_bar.prior_win.activateWindow()
                        self.menu_bar.prior_win.show()
                    else:
                        return
                elif type(test_result) != np.ndarray or test_result.shape != (2,):
                    warning = QMessageBox(self)
                    warning.setWindowTitle("WARNING")
                    warning.setIcon(QMessageBox.Warning)
                    warning.setText("Return value must be vectorized (1 x len(X)) and of type `np.array`. Press 'OK' if you would like to make changes, otherwise the prior mean model will be discarded.")
                    warning.setStandardButtons(QMessageBox.Ok)
                    
                    if warning.exec() == QMessageBox.Ok:
                        self.menu_bar.prior_win.setWindowState(self.menu_bar.prior_win.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)  # restoring to normal state
                        self.menu_bar.prior_win.activateWindow()
                        self.menu_bar.prior_win.show()
                    else:
                        return
            

                if self.GPBO:
                    prior = prior_mean_model_wrapper(custom_function, decision_transformer=self.GPBO.unnormalize)
                    for param in prior.parameters():
                        param.requires_grad = False
                    self.GPBO.prior_mean_model = prior
                else:
                    self.prior_mean_model = custom_function
                
            except Exception as exc:
                print(exc)
                critical = QMessageBox(self)
                critical.setWindowTitle('ERROR')
                critical.setText(f'Function crashed: {type(exc)}')
                critical.setDetailedText(str(exc))
                critical.setIcon(QMessageBox.Critical)
                critical.show()

                if critical.exec() == QMessageBox.Ok:
                    critical.close()
                    self.menu_bar.prior_win.setWindowState(self.menu_bar.prior_win.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)  # restoring to normal state
                    self.menu_bar.prior_win.activateWindow()
                    self.menu_bar.prior_win.show()
        def onHeaderChange(table: QTableWidget, column: int, text: str) -> None:
            """
            Since headers can be renamed, adjusts headers universally.
            """
            HORIZONTAL_TABLES = [self.main.x_table, self.main.boundry_table, self.main.candidate_pnt_table, self.main.pending_pnt_table]
            
            self.main.blockWidgetSignals(True)
            self.plots.blockWidgetSignals(True)

            HEADER = table.get_horizontal_labels()
            
            for tbl in HORIZONTAL_TABLES:
                if tbl is not table:
                    tbl.setHorizontalHeaderLabels(HEADER)

            if self.centralWidget().tabs.isTabEnabled(1):
                self.plots.acq_x_combobox.clear()
                self.plots.acq_x_combobox.addItems(HEADER)
                self.plots.acq_x_combobox.setCurrentIndex(self.plots.acq_x_combobox.previousIndex() if self.plots.acq_x_combobox.previousIndex() else 0)
                
                self.plots.acq_y_combobox.clear()
                self.plots.acq_y_combobox.addItems(HEADER)
                self.plots.acq_y_combobox.setCurrentIndex(self.plots.acq_y_combobox.previousIndex() if self.plots.acq_y_combobox.previousIndex() else 1)

                self.plots.post_x_combobox.clear()
                self.plots.post_x_combobox.addItems(HEADER)
                self.plots.post_x_combobox.setCurrentIndex(self.plots.post_x_combobox.previousIndex() if self.plots.post_x_combobox.previousIndex() else 0)
                
                self.plots.post_y_combobox.clear()
                self.plots.post_y_combobox.addItems(HEADER)
                self.plots.post_y_combobox.setCurrentIndex(self.plots.post_y_combobox.previousIndex() if self.plots.post_y_combobox.previousIndex() else 1)

                self.plots.acq_fixed_table.setVerticalHeaderLabels(HEADER)
                self.plots.post_fixed_table.setVerticalHeaderLabels(HEADER)

            self.plots.blockWidgetSignals(False)
            self.main.blockWidgetSignals(False)
        def onHorizontalScrollBarChange(value: int) -> None:
            """Synchronizes horizontal scroll bar values."""
            self.main.x_table.horizontalScrollBar().setValue(value)
            self.main.y_table.horizontalScrollBar().setValue(value)
            self.main.boundry_table.horizontalScrollBar().setValue(value)
            self.main.candidate_pnt_table.horizontalScrollBar().setValue(value)
            self.main.pending_pnt_table.horizontalScrollBar().setValue(value)
        def onUpdatedGPBO(epoch: int) -> None:
            """What must be done after the GPBO is updated."""
            print(f"Current Epoch: {epoch}")
            
            self.plot(self.main)
            
            self.main.epoch_label.setText(str(self.GPBO.epoch()))
            self.plots.epoch_spinbox.setMaximum(epoch)
            self.main.batch_spin_box.setEnabled(True)
            
            if not self.centralWidget().tabs.isTabEnabled(1):
                self.startup.emit()
        def onPlotStarted() -> None:
            """
            What must be done while the plotting occurs.
            """
            self.main.progress_button.enableProgressBar()
            self.plots.progress_button.enableProgressBar()
        def onPlotFinished(canvas_code: int) -> None:
            """
            What must be done after the plotting is finished.
            """
            if canvas_code == self.main.CODE:
                canvas = self.main.canvas
                epoch = self.GPBO.epoch()
                
                # AXES VISIBILITY
                if self.main.plot_button.isChecked():
                    if epoch < 2:
                        canvas.acquisition_ax.set_visible(True)
                        canvas.posterior_ax.set_visible(True)
                        canvas.obj_history_ax.set_visible(False)
                        canvas._get_obj_twinx().set_visible(False)
                    else:
                        canvas.acquisition_ax.set_visible(True)
                        canvas.posterior_ax.set_visible(True)
                        canvas.obj_history_ax.set_visible(True)
                        canvas._get_obj_twinx().set_visible(True)
                else:
                    if epoch > 1:
                        canvas.acquisition_ax.set_visible(False)
                        canvas.posterior_ax.set_visible(False)
                        canvas.obj_history_ax.set_visible(True)
                        canvas._get_obj_twinx().set_visible(True)
                    else:
                        canvas.acquisition_ax.set_visible(False)
                        canvas.posterior_ax.set_visible(False)
                        canvas.obj_history_ax.set_visible(False)
                        canvas._get_obj_twinx().set_visible(False)
                        
            elif canvas_code == self.plots.CODE:
                canvas = self.plots.canvas
                epoch = self.plots.epoch_spinbox.value()
                
                # AXES VISIBLITY
                if epoch < 2:
                    canvas.acquisition_ax.set_visible(True)
                    canvas.posterior_ax.set_visible(True)
                    canvas.obj_history_ax.set_visible(False)
                    canvas._get_obj_twinx().set_visible(False)
                else:
                    canvas.acquisition_ax.set_visible(True)
                    canvas.posterior_ax.set_visible(True)
                    canvas.obj_history_ax.set_visible(True)
                    canvas._get_obj_twinx().set_visible(True)
            else:
                raise ValueError(f"{canvas_code} is an invalid canvas code.")
                    
            # LABELS
            DECISION_HEADER = self.main.x_table.get_horizontal_labels()
            acq_labels = (self.plots.acq_x_combobox.currentText(), self.plots.acq_y_combobox.currentText()) if self.GPBO.epoch() > 1 else (DECISION_HEADER[0], DECISION_HEADER[1])
            post_labels = (self.plots.post_x_combobox.currentText(), self.plots.post_y_combobox.currentText()) if self.GPBO.epoch() > 1 else (DECISION_HEADER[0], DECISION_HEADER[1])
            
            canvas.format(acq_labels, post_labels)
            canvas.reload()
            
            self.main.query_candidates_button.setEnabled(True)
        def onQueryStarted() -> None:
            """
            What must be done while the query is running.
            """
            self.main.update_gp_button.setEnabled(False)
            self.main.query_candidates_button.setEnabled(False)
        def onQueryFinished(candidates: List[List[float]]) -> None:
            """
            What must be done after the query candidates are recieved.
            """
            self.main.candidate_pnt_table.setRowCount(0)  # pseudo-clear
            self.main.candidate_pnt_table.setRowCount(len(candidates))
            self.main.candidate_pnt_table.fill(candidates)
            
            if self.plots.epoch_spinbox.value() == self.GPBO.epoch():
                candidate_pnts = self.GPBO.history[self.GPBO.epoch()]['x1']

                for i in range(1, self.plots.query_combobox.count()):
                    self.plots.query_combobox.removeItem(1)
                
                for i in range(len(candidate_pnts)):
                    self.plots.query_combobox.addItem(f'{i+1}')
            
            self.main.update_gp_button.setEnabled(True)
            self.main.query_candidates_button.setEnabled(True)
            
            if self.main.plot_button.isChecked():
                self.plot(self.main)
        def onUpdateGPRequest() -> None:
            """
            What occurs when a update is request on the GPBO model.
            """
            try:
                self.update_GP()
                self.main.candidate_pnt_table.setRowCount(0)
                self.main.pending_pnt_table.setRowCount(0)
            except Exception as exc:
                traceback.print_exc()
                QMessageBox.critical(self, "CRITICAL", str(exc))
        def onStartup() -> None:
            """
            What occurs on the transition from model initializiation to GP updates.
            """
            self.main.x_table.enableColumnChanges(False)
            self.main.boundry_table.enableColumnChanges(False)
            
            self.main.candidate_pnt_table.setEnabled(True)
            self.main.pending_pnt_table.setEnabled(True)
            
            # Migrating into "Plots" tab
            FIXED_HEADER = ["Fixed Value"]
            DECISION_HEADER = self.main.x_table.get_horizontal_labels()
            FIXED_VAL = 0.0
            NUM_DIMENSIONS = len(DECISION_HEADER)

            self.plots.blockWidgetSignals(True)

            self.plots.epoch_spinbox.setValue(1)
            self.plots.epoch_spinbox.setMinimum(1)
            self.plots.epoch_spinbox.setMaximum(1)
            self.plots.acq_min_button.setChecked(True)
            self.plots.post_min_button.setChecked(True)

            self.plots.acq_x_combobox.clear()
            self.plots.acq_x_combobox.addItems(DECISION_HEADER)
            self.plots.acq_x_combobox.setCurrentIndex(0)
            self.plots.acq_x_combobox.setPreviousIndex(0)
            
            self.plots.acq_y_combobox.clear()
            self.plots.acq_y_combobox.addItems(DECISION_HEADER)
            self.plots.acq_y_combobox.setCurrentIndex(1)
            self.plots.acq_y_combobox.setPreviousIndex(1)

            self.plots.post_x_combobox.clear()
            self.plots.post_x_combobox.addItems(DECISION_HEADER)
            self.plots.post_x_combobox.setCurrentIndex(0)
            self.plots.post_x_combobox.setPreviousIndex(0)
            
            self.plots.post_y_combobox.clear()
            self.plots.post_y_combobox.addItems(DECISION_HEADER)
            self.plots.post_y_combobox.setCurrentIndex(1)
            self.plots.post_y_combobox.setPreviousIndex(1)

            for tbl in [self.plots.acq_fixed_table, self.plots.post_fixed_table]:
                tbl.clear()
                tbl.setColumnCount(1)
                tbl.setRowCount(NUM_DIMENSIONS)
                tbl.setHorizontalHeaderLabels(FIXED_HEADER)
                tbl.setVerticalHeaderLabels(DECISION_HEADER)
                tbl.setEnabled(False)
                for i in range(NUM_DIMENSIONS):
                    item = CenteredTableItem(str(FIXED_VAL))
                    tbl.setItem(i, 0, item)
                    if i < 2:
                        tbl.setRowHidden(i, True)

            self.plots.query_combobox.clear()
            self.plots.query_combobox.addItem("All")

            self.plots.blockWidgetSignals(False)
            
            self.centralWidget().tabs.setTabEnabled(self.plots.CODE, True)
        
        
        # CONNECTIONS
        self.startup.connect(onStartup)
        self.updatedGPBO.connect(onUpdatedGPBO)
        self.plotStarted.connect(onPlotStarted)
        self.plotFinished.connect(onPlotFinished)
        self.queryStarted.connect(onQueryStarted)
        self.queryFinished.connect(onQueryFinished)
        
        self.main.update_gp_button.clicked.connect(onUpdateGPRequest)
        self.main.query_candidates_button.clicked.connect(self.query)
        self.plots.update_proj_button.clicked.connect(lambda: self.plot(self.plots))
        
        self.menu_bar.newFile.connect(onNewFileRequest)
        self.menu_bar.openFile.connect(onOpenFileRequest)
        self.menu_bar.refreshPlots.connect(onPlotRefreshRequest)
        self.menu_bar.saveAsFile.connect(onSaveFileAsRequest)
        self.menu_bar.preferences.connect(onOpenPreferencesRequest)
        
        self.main.x_table.headerChanged.connect(lambda col, text: onHeaderChange(self.main.x_table, col, text))
        self.main.boundry_table.headerChanged.connect(lambda col, text: onHeaderChange(self.main.boundry_table, col, text))

        self.main.x_table.dimensionsChanged.connect(lambda code, index: onTableDimensionsChanged(self.main.x_table, code, index))
        self.main.y_table.dimensionsChanged.connect(lambda code, index: onTableDimensionsChanged(self.main.y_table, code, index))
        self.main.boundry_table.dimensionsChanged.connect(lambda code, index: onTableDimensionsChanged(self.main.boundry_table, code, index))

        self.main.candidate_pnt_table.selectedToDecisionRequested.connect(lambda: movePoints(self.main.candidate_pnt_table, self.main.x_table, selected=True))
        self.main.candidate_pnt_table.allToDecisionRequested.connect(lambda: movePoints(self.main.candidate_pnt_table, self.main.x_table))
        self.main.candidate_pnt_table.selectedToPendingRequested.connect(lambda: movePoints(self.main.candidate_pnt_table, self.main.pending_pnt_table, selected=True))
        self.main.candidate_pnt_table.allToPendingRequested.connect(lambda: movePoints(self.main.candidate_pnt_table, self.main.pending_pnt_table))

        self.main.x_table.verticalScrollBar().valueChanged.connect(self.main.y_table.verticalScrollBar().setValue)
        self.main.x_table.horizontalScrollBar().valueChanged.connect(onHorizontalScrollBarChange)
        self.main.y_table.verticalScrollBar().valueChanged.connect(self.main.x_table.verticalScrollBar().setValue)
        self.main.candidate_pnt_table.horizontalScrollBar().valueChanged.connect(onHorizontalScrollBarChange)
        self.main.pending_pnt_table.horizontalScrollBar().valueChanged.connect(onHorizontalScrollBarChange)
        self.main.boundry_table.horizontalScrollBar().valueChanged.connect(onHorizontalScrollBarChange)

        self.main.canvas.progressUpdate.connect(self.main.progress_button.updateValue)
        self.main.canvas.progressUpdate.connect(self.plots.progress_button.updateValue)
        self.plots.canvas.progressUpdate.connect(self.main.progress_button.updateValue)
        self.plots.canvas.progressUpdate.connect(self.plots.progress_button.updateValue)

        self.main.ucb_button.toggled.connect(lambda state: self.main.beta_spinbox.setEnabled(state))
        
        self.main.obj_func_btn.clicked.connect(self.main.obj_func_win.show)
        self.main.obj_func_win.gotCustomFunction.connect(onGotObjectiveFunction)
        self.menu_bar.priorRequested.connect(self.menu_bar.prior_win.show)
        self.menu_bar.prior_win.gotCustomFunction.connect(onGotPriorFunction)

        self.plots.acq_fix_button.toggled.connect(lambda state: self.plots.acq_fixed_table.setEnabled(state))
        self.plots.post_fix_button.toggled.connect(lambda state: self.plots.post_fixed_table.setEnabled(state))
        self.plots.acq_x_combobox.currentIndexChanged.connect(lambda: onPlotsComboBoxChange(self.plots.acq_x_combobox, self.plots.acq_y_combobox))
        self.plots.acq_y_combobox.currentIndexChanged.connect(lambda: onPlotsComboBoxChange(self.plots.acq_y_combobox, self.plots.acq_x_combobox))
        self.plots.post_x_combobox.currentIndexChanged.connect(lambda: onPlotsComboBoxChange(self.plots.post_x_combobox, self.plots.post_y_combobox))
        self.plots.post_y_combobox.currentIndexChanged.connect(lambda: onPlotsComboBoxChange(self.plots.post_y_combobox, self.plots.post_x_combobox)) 

        self.plots.epoch_spinbox.valueChanged.connect(onPlotsEpochChange)

        self.menu_bar.pref_win.app_font_sb.valueChanged.connect(app_font_size_change)
        self.menu_bar.pref_win.path_log_le.editingFinished.connect(lambda: verify_path(self.menu_bar.pref_win.path_log_le))

    def update_GP(self) -> None:
        x = self.main.x_table.to_list()
        y = self.main.y_table.to_list()
        
        list1D_all_none = lambda list1D: all(x is None for x in list1D)
        list1D_any_none = lambda list1D: any(x is None for x in list1D)
        
        possible_mistakes = []
        remaining_x = []
        remaining_y = []
        for i in range(len(x)):
            if list1D_all_none(x[i]) and list1D_all_none(y[i]):
                continue
            elif list1D_any_none(x[i]) or list1D_any_none(y[i]):
                possible_mistakes.append(i)
            else:
                remaining_x.append(x[i])
                remaining_y.append(y[i])
                
        if len(possible_mistakes) > 0:
            if len(possible_mistakes) == 1:
                 detailed_text =  "Row: " + str(possible_mistakes[0] + 1)
            elif len(possible_mistakes) == 2:
                detailed_text = "Rows: " + str(possible_mistakes[0] + 1) + " and " + str(possible_mistakes[1] + 1)
            else:
                detailed_text = ", ".join(str(i + 1) for i in possible_mistakes[:-1])
                detailed_text += ", and " + str(possible_mistakes[-1] + 1)
                detailed_text = "Rows: " + detailed_text
                
            warning = QMessageBox(QMessageBox.Warning, "WARNING", "Some data rows are incomplete. Would you like to proceed?", QMessageBox.No | QMessageBox.Yes, self.control)
            warning.setDetailedText(detailed_text)
            if not warning.exec() == QMessageBox.Yes:
                raise ValueError("GPBO cancelled.")
            
        x = np.array(remaining_x)
        y = np.array(remaining_y)
        
        if x.size == 0 or y.size == 0:
            raise ValueError("The GPBO needs data to be initialized.")
            
        if not hasattr(self, "GPBO") or not self.GPBO:
            bounds = np.array(self.main.boundry_table.to_list())
            batch_size = self.main.batch_spin_box.value()
            
            if self.main.ucb_button.isChecked():
                acq_func = "UCB"
            else:
                raise ValueError("Please select an acquisition function.")
            self.GPBO = interactiveGPBO(x, y, bounds=bounds, batch_size=batch_size, acquisition_func=acq_func, prior_mean_model=self.prior_mean_model)
        else:
            self.GPBO.update_GP(x=x, y=y)
        self.updatedGPBO.emit(self.GPBO.epoch())
    def plot(self, view: Union[MainView, PlotsView]) -> None:
        if not hasattr(self, "GPBO") or not self.GPBO:
            raise AttributeError("The GPBO has not yet been initialized.")
        elif view is self.main:
            query_number = len(self.GPBO.history[self.GPBO.epoch()]["x1"]) - 1 if len(self.GPBO.history[self.GPBO.epoch()]["x1"]) - 1 > -1 else None
            epoch = self.GPBO.epoch()
        elif view is self.plots:
            query_number = int(self.plots.query_combobox.currentText()) - 1 if self.plots.query_combobox.currentText() != "All" else None
            epoch = self.plots.epoch_spinbox.value()
        else:
            raise ValueError(f"Cannot plot at {view}.")
        
        DECISION_HEADER = self.main.x_table.get_horizontal_labels()
        
        # LABELS
        acq_xlabel = self.plots.acq_x_combobox.currentText() if self.plots.acq_x_combobox.currentIndex() != -1 else DECISION_HEADER[0]
        acq_ylabel = self.plots.acq_y_combobox.currentText() if self.plots.acq_x_combobox.currentIndex() != -1 else DECISION_HEADER[1]
        post_xlabel = self.plots.post_x_combobox.currentText() if self.plots.acq_x_combobox.currentIndex() != -1 else DECISION_HEADER[0]
        post_ylabel = self.plots.post_y_combobox.currentText() if self.plots.acq_x_combobox.currentIndex() != -1 else DECISION_HEADER[1]
        
        # DIMENSION INDICES
        acq_xdim = self.plots.acq_x_combobox.currentIndex() if self.plots.acq_x_combobox.currentIndex() != -1 else DECISION_HEADER.index(acq_xlabel)
        acq_ydim = self.plots.acq_y_combobox.currentIndex() if self.plots.acq_x_combobox.currentIndex() != -1 else DECISION_HEADER.index(acq_ylabel)
        post_xdim = self.plots.post_x_combobox.currentIndex() if self.plots.acq_x_combobox.currentIndex() != -1 else DECISION_HEADER.index(post_xlabel)
        post_ydim = self.plots.post_y_combobox.currentIndex() if self.plots.acq_x_combobox.currentIndex() != -1 else DECISION_HEADER.index(post_ylabel)
                
        # PROJECTION TYPE
        if self.GPBO.epoch() < 1 or self.plots.acq_min_button.isChecked():
            acq_proj_min = True
            acq_proj_mean = False
            acq_fixed_vals = None
        elif self.plots.acq_mean_button.isChecked():
            acq_proj_min = True
            acq_proj_mean = False
            acq_fixed_vals = None
        elif self.plots.acq_fix_button.isChecked():
            acq_proj_min = False
            acq_proj_mean = False
            acq_fixed_vals = self.plots.acq_fixed_table.to_dict()
        if self.GPBO.epoch() < 1 or self.plots.post_min_button.isChecked():
            post_proj_min = True
            post_proj_mean = False
            post_fixed_vals = None
        elif self.plots.acq_mean_button.isChecked():
            post_proj_min = False
            post_proj_mean = True
            post_fixed_vals = None
        elif self.plots.acq_fix_button.isChecked():
            post_proj_min = False
            post_proj_mean = False
            post_fixed_vals = self.plots.acq_fixed_table.to_dict()

        # KWARGS
        beta = self.main.beta_spinbox.value() if self.main.ucb_button.isChecked() else None
        
        def plot():
            view.canvas.clear()
            
            if self.main.plot_button.isChecked():
                self.plotStarted.emit()
                try:
                    view.canvas.plot_acquisition(self.GPBO, epoch, query_number, beta, acq_xdim, acq_ydim, acq_proj_min, acq_proj_mean, acq_fixed_vals)
                    view.canvas.plot_posterior(self.GPBO, epoch, query_number, post_xdim, post_ydim, post_proj_min, post_proj_mean, post_fixed_vals)
                except RuntimeError:
                    view.canvas.plot_acquisition(self.GPBO, epoch, query_number, beta, acq_xdim, acq_ydim, acq_proj_min, acq_proj_mean, acq_fixed_vals, overdrive=True)
                    view.canvas.plot_posterior(self.GPBO, epoch, query_number, post_xdim, post_ydim, post_proj_min, post_proj_mean, post_fixed_vals, overdrive=True)
            view.canvas.plot_obj_history(self.GPBO)  # fast
            self.plotFinished.emit(0 if view.canvas is self.main.canvas else 1 if view.canvas is self.plots.canvas else None)
                
        thread = threading.Thread(target=plot)
        thread.start()
    def query(self) -> None:
        self.queryStarted.emit()
        
        # PENDING POINTS
        pending_pnts = self.main.pending_pnt_table.to_list()
        
        list1D_all_none = lambda list1D: all(x is None for x in list1D)
        list1D_any_none = lambda list1D: any(x is None for x in list1D)
        
        possible_mistakes = []
        remaining = []
        for i in range(len(pending_pnts)):
            if list1D_all_none(pending_pnts[i]):
                continue
            elif list1D_any_none(pending_pnts[i]):
                possible_mistakes.append(i)
            else:
                remaining.append(pending_pnts[i])

        if len(possible_mistakes) > 0:
            if len(possible_mistakes) == 1:
                 detailed_text =  "Row: " + str(possible_mistakes[0] + 1)
            elif len(possible_mistakes) == 2:
                detailed_text = "Rows: " + str(possible_mistakes[0] + 1) + " and " + str(possible_mistakes[1] + 1)
            else:
                detailed_text = ", ".join(str(i + 1) for i in possible_mistakes[:-1])
                detailed_text += ", and " + str(possible_mistakes[-1] + 1)
                detailed_text = "Rows: " + detailed_text
                
            warning = QMessageBox(QMessageBox.Warning, "WARNING", "Some pending points are incomplete. Would you like to proceed?", QMessageBox.No | QMessageBox.Yes, self.control)
            warning.setDetailedText(detailed_text)
            if not warning.exec() == QMessageBox.Yes:
                self.queryFinished.emit(None)
                raise ValueError("GPBO cancelled.")
            
        pending_pnts = remaining if remaining else None
        
        if self.main.ucb_button.isChecked():
            acq_func = qUpperConfidenceBound
            acq_args = {"beta": self.main.beta_spinbox.value()}
        
        def query():
            candidates = self.GPBO.query_candidates(batch_size=self.main.batch_spin_box.value(), X_pending=pending_pnts, acquisition_func=acq_func, acquisition_func_args=acq_args)
            # this is where I would order by ramping rate
            self.queryFinished.emit(candidates.tolist())
        thread = threading.Thread(target=query)
        thread.start()
