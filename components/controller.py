import threading
from importlib import import_module
from typing import List, Tuple, Union, Dict
import pickle
import json
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
        FIXED_VAL: float = 0.0
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

        if hasattr(self, "colorbars"):
            for cbar in self.colorbars.values():
                cbar.remove()
            self.colorbars.clear()
        else:
            self.colorbars = {}
            
        self.centralWidget().tabs.setCurrentIndex(0)  # `0` = "Main"
        self.centralWidget().tabs.setTabEnabled(1, False)  # `1` = "Plots"

        # Main Page
        self.main.blockWidgetSignals(True)
        
        self.main.epoch_label.setText(str(EPOCH))
        self.main.plot_button.setChecked(False)
        self.main.ucb_button.setChecked(True)
        self.main.batch_spin_box.setValue(BATCH_SIZE)
        self.main.batch_spin_box.setRange(1, 100)
        self.main.beta_spinbox.setValue(BETA)
        self.main.beta_spinbox.setEnabled(True)
        
        self.main.candidate_pnt_table.clear()
        self.main.candidate_pnt_table.setRowCount(0)
        self.main.candidate_pnt_table.setColumnCount(NUM_DIMENSIONS)
        self.main.candidate_pnt_table.setHorizontalHeaderLabels(DECISION_HEADER)
        
        self.main.pending_pnt_table.clear()
        self.main.pending_pnt_table.setRowCount(0)
        self.main.pending_pnt_table.setColumnCount(NUM_DIMENSIONS)
        self.main.pending_pnt_table.setHorizontalHeaderLabels(DECISION_HEADER)
        
        self.main.x_table.clear()
        self.main.x_table.setColumnCount(NUM_DIMENSIONS)
        self.main.x_table.setRowCount(ROW_COUNT)
        self.main.x_table.setHorizontalHeaderLabels(DECISION_HEADER)
            
        self.main.y_table.clear()
        self.main.y_table.setColumnCount(1)
        self.main.y_table.setRowCount(ROW_COUNT)
        self.main.y_table.setHorizontalHeaderLabels(OBJECTIVE_HEADER)

        self.main.boundry_table.clear()
        self.main.boundry_table.setColumnCount(NUM_DIMENSIONS)
        self.main.boundry_table.setRowCount(2)
        self.main.boundry_table.setHorizontalHeaderLabels(DECISION_HEADER)
        self.main.boundry_table.setVerticalHeaderLabels(BOUNDRY_HEADER)
        
        self.main.canvas.hide_axes()
        self.main.canvas.draw_idle()

        self.main.blockWidgetSignals(False)
    def set_connections(self) -> None:
        
        # LOGIC
        def data_table_cell_change(table: Union[XTable, YTable], row: int):
            pass
        def showBaseContextMenu(table: Table, pos: QPoint) -> None:
            menu = TableContextMenu(parent=table)
            menu.popup(table.viewport().mapToGlobal(pos))
        def showStandardContextMenu(table: Union[XTable, YTable], pos: QPoint):
            menu = StandardTableContextMenu(pos, parent=table)
            if self.centralWidget().tabs.isTabEnabled(1):
                menu.insert_column_action.setVisible(False)
                menu.remove_column_action.setVisible(False)
            menu.popup(table.viewport().mapToGlobal(pos))
        def showHeaderContextMenu(table: Union[XTable, YTable], pos: QPoint):
            column: int = table.horizontalHeader().logicalIndexAt(pos)
            menu = HeaderContextMenu(column, parent=table)
            menu.popup(table.horizontalHeader().viewport().mapToGlobal(pos))
        def showQueryContextMenu(table: CandidatePendingTableContextMenu, pos: QPoint):
            menu = CandidatePendingTableContextMenu(pos, parent=table)
            menu.popup(table.viewport().mapToGlobal(pos))
        def movePoints(src: Table, dest: Table, selected: bool = False):
            src.blockSignals(True)
            dest.blockSignals(True)

            items = src.take_items(selected)
            if isinstance(src, PendingTable) or isinstance(src, CandidateTable):
                src.remove_empty_rows(True)

            dest.append_items(items)
            
            # Handle y table automatic row addition
            if isinstance(dest, XTable):
                self.main.y_table.blockSignals(True)
                self.main.y_table.setRowCount(dest.rowCount())
                self.main.y_table.blockSignals(False)
            
            # Allow the signal to operate again
            dest.blockSignals(False)
            src.blockSignals(False)
        def updateGP():
            try:
                if not self.GPBO:
                    self.GPBO = SuperGPBO(self)#, prior_mean_model=self.custom_function)
                self.GPBO.update()
                
                if self.main.epoch_label.text() == 1:
                    self.centralWidget().tabs.setTabEnabled(1, True)
            except Exception as exc:
                print(exc)
                QMessageBox.critical(self, "CRITICAL", str(exc), QMessageBox.Ok)
        def applyObjFuncToY():
            self.main.y_table.blockSignals(True)

            glbs = {}
            lcls = {}

            # Format string as a pseudo-function
            func_str = self.main.obj_func_win.text_edit.toPlainText()
            func_str = "def function(X):\n\t" + func_str.replace('\n', '\n\t')
            
            try:
                exec(func_str, glbs, lcls)
                custom_function = list(lcls.values())[0]

                # Get decision parameters (account for extra row)
                list1D_any_none = lambda list1D: any(x is None for x in list1D)
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
        def onPlotsComboBoxChange(src: MemoryComboBox, other: MemoryComboBox):
            print("Plots ComboBox Change...")

            src.blockSignals(True)
            other.blockSignals(True)
                        
            prev = src.previousIndex()

            if src.currentIndex() == other.currentIndex():
                print("\tSwapping ComboBoxes")
                other.setPreviousIndex(prev)
                other.setCurrentIndex(prev)

            src.setPreviousIndex(src.currentIndex())

            other.blockSignals(False)
            src.blockSignals(False)
        def onPlotsEpochChange(epoch: Union[str, int]):
            if not self.GPBO:
                return
            
            print("Plots Epoch Change...")
            epoch = int(epoch)
            candidate_pnts = self.GPBO.history[epoch]["x1"]

            print("\tQuery ComboBox")
            for i in range(1, self.plots.query_combobox.count()):
                print(f"\t\tRemove Query {i}")
                self.plots.query_combobox.removeItem(1)

            for i in range(len(candidate_pnts)):
                print(f"\t\tAdd Query {i}")
                self.plots.query_combobox.addItem(f"{i + 1}")
        def on_new_file():
            self.set_initial_states()
        def on_open_file():
            while True:
                filename = QFileDialog.getOpenFileName(self, 'Open File', filter="All Files (*.*);;JSON File (*.json);;PICKLE File (*.pickle)")[0]  # previously tuple
                if filename:
                    _, extension = os.path.splitext(filename)
                    if extension != ".pickle" and extension != ".json":
                        warning = QMessageBox()
                        warning.setIcon(QMessageBox.Critical)
                        warning.setText("Didn't select a *.pickle or *.json file")
                        warning.setWindowTitle("ERROR")
                        warning.setStandardButtons(QMessageBox.Ok)

                        if warning.exec() == QMessageBox.Ok:
                            warning.close()
                            continue
                break

            if not filename:
                return
            
            self.set_initial_states()

            self.GPBO = SuperGPBO(self, filename=filename)
            
            num_dimensions = self.GPBO.dim
            
            # Fill arrays using X and Y data
            self.main.x_table.fill(self.GPBO.x)
            self.main.y_table.fill(self.GPBO.y)
            
            # Rotate array to fit bounds table and fill
            bounds = np.rot90(self.GPBO.bounds)
            self.main.boundry_table.fill(bounds)

            # Get candidates from GPBO.history[-1]['x1'] to put into candidate points
            
            # Get pending from GPBO.history[-1]['X_pending'] to put into pending points
        
            
            # Reset initialization table labels
            x_labels = [f'x[{i}]' for i in range(num_dimensions)]
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


            if self.main.plot_button.isChecked():
                def plot():
                    self.GPBO.plot(preview=True)
                    self.centralWidget().tabs.setTabEnabled(1, True)
                thread = threading.Thread(target=plot)
                thread.start()
        def on_plot_refresh():
            if self.GPBO:
                self.GPBO.refresh_plots()
        def on_save_as():
            GPBO = self.GPBO

            filename = QFileDialog.getSaveFileName(self, "Save As", filter="All Files (*.*);;JSON File (*.json);;PICKLE File (*.pickle)")[0]  # previously tuple
            path, tag = os.path.split(filename)
            path += '/' if not path.endswith('/') else ''
            
            # From GPBO.write_log()...
            data = []
            for hist in GPBO.history:
                hist_ = {}
                for key, val in hist.items():
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
                        if type(val) is not list:  # new
                            val = val.tolist()
                        hist_[key] = val
                data.append(hist_)
            
            if tag.endswith(".pickle"):
                with open(path + tag, "wb") as file:
                    pickle.dump(data, file)
            elif tag.endswith(".json"):
                with open(path + tag, "w") as file:
                    json.dump(data, file)
        def on_open_preferences():
            self.menu_bar.pref_win.show()
        def app_font_size_change(size: int):
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
        def table_column_inserted(table: XTable, column: int) -> None:
            # Automatically format default header items
            for i, label in enumerate(table.get_horizontal_labels()):
                if not label or label[:2] + label[-1] == "x[]":
                    item = QTableWidgetItem(f"x[{i}]")
                    table.setHorizontalHeaderItem(i, item)

            labels = table.get_horizontal_labels()
            tables = [self.main.boundry_table, self.main.candidate_pnt_table, self.main.pending_pnt_table]
            for tbl in tables:
                tbl.insertColumn(column)
                tbl.setHorizontalHeaderLabels(labels)
                tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        def table_column_removed(table: XTable, column: int) -> None:
            # Automatically format default header items
            for i, label in enumerate(table.get_horizontal_labels()):
                if not label or label[:2] + label[-1] == "x[]":
                    item = QTableWidgetItem(f"x[{i}]")
                    table.setHorizontalHeaderItem(i, item)
                    
            labels = table.get_horizontal_labels()
            tables = [self.main.boundry_table, self.main.candidate_pnt_table, self.main.pending_pnt_table]
            for tbl in tables:
                tbl.removeColumn(column)
                tbl.setHorizontalHeaderLabels(labels)
                tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        def table_row_inserted(table: Union[XTable, YTable], row: int) -> None:
            if isinstance(table, XTable):
                self.main.y_table.insertRow(row)
            elif isinstance(table, YTable):
                self.main.x_table.insertRow(row)
        def table_row_removed(table: Union[XTable, YTable], row: int) -> None:
            if isinstance(table, XTable):
                self.main.y_table.removeRow(row)
            elif isinstance(table, YTable):
                self.main.x_table.removeRow(row)
        def verify_cell_exists(table: Union[XTable, YTable, BoundryTable]) -> None:
            print(table.currentItem())
            if not table.currentItem() or not isinstance(table.currentItem(), CenteredTableItem):
                table.setItem(table.currentRow(), table.currentColumn(), CenteredTableItem())
        def on_tab_enabled(tab: int, b: bool) -> None:
        
            MAIN = 0
            PLOTS = 1
            
            if tab == PLOTS and b is True:
                self.plots.blockWidgetSignals(True)

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

                fix_tbls = [self.plots.acq_fixed_table, self.plots.post_fixed_table]
                for tbl in fix_tbls:
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
        def queryGP():
            if self.GPBO:
                self.main.query_candidates_button.setEnabled(False)
                self.GPBO.query()
        def on_epoch_change(epoch: int):
            print(f"Epoch Changed To... {epoch}")
            self.plots.epoch_spinbox.setMaximum(epoch)
            if not self.centralWidget().tabs.isTabEnabled(1):
                self.centralWidget().tabs.setTabEnabled(1, True)
        def on_progress_bar_enable(b: bool):
            if b is True:
                self.main.query_candidates_button.setEnabled(False)
            elif b is False:
                self.main.query_candidates_button.setEnabled(True)
        def got_prior():
            glbs = {}
            lcls = {}

            # Format string as a pseudo-function
            func_str = self.menu_bar.prior_win.text_edit.toPlainText()
            func_str = "def function(X):\n\t" + func_str.replace('\n', '\n\t')
            
            try:
                exec(func_str, glbs, lcls)
                custom_function = list(lcls.values())[0]
            
                # self.GPBO.prior_mean_model = prior_mean_model_wrapper(custom_function, decision_transformer=self.GPBO.unnormalize)
                # self.GPBO.prior_mean_model = custom_function
                self.custom_function = custom_function

                bounds = np.array(self.main.boundry_table.to_list())
                
                x = np.random.rand(2, 4)
                print(x)
                print(custom_function(x))
                # print(X.shape, type(X), X.dtype)
                # x = np.array(X)
                # x1 = np.sum(x**2,axis=1)
                # x2 = prior_levy(X)
                # print(type(x1), x1.shape, x1.dtype)
                # print(type(x2), x2.shape, x2.dtype)
                # a = np.random.rand(2, len(bounds))
                # a * bounds[1,:] - bounds[0,:] + bounds[0,:]
                

                
            except Exception as exc:
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
        def onHeaderChange(column: int, text: str):
            self.main.blockWidgetSignals(True)
            self.plots.blockWidgetSignals(True)

            DECISION_HEADER = self.main.x_table.get_horizontal_labels()
            
            self.main.boundry_table.setHorizontalHeaderLabels(DECISION_HEADER)
            self.main.candidate_pnt_table.setHorizontalHeaderLabels(DECISION_HEADER)
            self.main.pending_pnt_table.setHorizontalHeaderLabels(DECISION_HEADER)

            if self.centralWidget().tabs.isTabEnabled(1):
                self.plots.acq_x_combobox.clear()
                self.plots.acq_x_combobox.addItems(DECISION_HEADER)
                self.plots.acq_x_combobox.setCurrentIndex(self.plots.acq_x_combobox.previousIndex() if self.plots.acq_x_combobox.previousIndex() else 0)
                
                self.plots.acq_y_combobox.clear()
                self.plots.acq_y_combobox.addItems(DECISION_HEADER)
                self.plots.acq_y_combobox.setCurrentIndex(self.plots.acq_y_combobox.previousIndex() if self.plots.acq_y_combobox.previousIndex() else 1)

                self.plots.post_x_combobox.clear()
                self.plots.post_x_combobox.addItems(DECISION_HEADER)
                self.plots.post_x_combobox.setCurrentIndex(self.plots.post_x_combobox.previousIndex() if self.plots.post_x_combobox.previousIndex() else 0)
                
                self.plots.post_y_combobox.clear()
                self.plots.post_y_combobox.addItems(DECISION_HEADER)
                self.plots.post_y_combobox.setCurrentIndex(self.plots.post_y_combobox.previousIndex() if self.plots.post_y_combobox.previousIndex() else 1)

                self.plots.acq_fixed_table.setVerticalHeaderLabels(DECISION_HEADER)
                self.plots.post_fixed_table.setVerticalHeaderLabels(DECISION_HEADER)

            self.plots.blockWidgetSignals(False)
            self.main.blockWidgetSignals(False)
        def onHorizontalScrollBarChange(value: int) -> None:
            self.main.x_table.horizontalScrollBar().setValue(value)
            self.main.y_table.horizontalScrollBar().setValue(value)
            self.main.boundry_table.horizontalScrollBar().setValue(value)
            self.main.candidate_pnt_table.horizontalScrollBar().setValue(value)
            self.main.pending_pnt_table.horizontalScrollBar().setValue(value)
            
        # CONNECTIONS
        self.main.x_table.cellChanged.connect(lambda row, col: data_table_cell_change(self.main.x_table, row))
        self.main.y_table.cellChanged.connect(lambda row, col: data_table_cell_change(self.main.y_table, row))
        self.main.x_table.headerChanged.connect(onHeaderChange)

        self.main.x_table.itemSelectionChanged.connect(lambda: verify_cell_exists(self.main.x_table))
        self.main.y_table.itemSelectionChanged.connect(lambda: verify_cell_exists(self.main.y_table))
        self.main.boundry_table.itemSelectionChanged.connect(lambda: verify_cell_exists(self.main.boundry_table))
        self.main.pending_pnt_table.itemSelectionChanged.connect(lambda: verify_cell_exists(self.main.pending_pnt_table))
        self.main.pending_pnt_table.currentItemChanged.connect(lambda: verify_cell_exists(self.main.pending_pnt_table))

        self.main.x_table.rowInserted.connect(lambda row: table_row_inserted(self.main.x_table, row))
        self.main.x_table.columnInserted.connect(lambda col: table_column_inserted(self.main.x_table, col))
        self.main.x_table.rowRemoved.connect(lambda row: table_row_removed(self.main.x_table, row))
        self.main.x_table.columnRemoved.connect(lambda col: table_column_removed(self.main.x_table, col))
        self.main.y_table.rowInserted.connect(lambda row: table_row_inserted(self.main.y_table, row))
        self.main.y_table.columnInserted.connect(lambda col: table_column_inserted(self.main.y_table, col))
        self.main.y_table.rowRemoved.connect(lambda row: table_row_removed(self.main.y_table, row))
        self.main.y_table.columnRemoved.connect(lambda col: table_column_removed(self.main.y_table, col))

        self.main.x_table.customContextMenuRequested.connect(lambda pos: showStandardContextMenu(self.main.x_table, pos))
        self.main.x_table.horizontalHeader().customContextMenuRequested.connect(lambda pos: showHeaderContextMenu(self.main.x_table, pos))
        self.main.y_table.customContextMenuRequested.connect(lambda pos: showStandardContextMenu(self.main.y_table, pos))
        self.main.y_table.horizontalHeader().customContextMenuRequested.connect(lambda pos: showHeaderContextMenu(self.main.y_table, pos))
        self.main.boundry_table.customContextMenuRequested.connect(lambda pos: showBaseContextMenu(self.main.boundry_table, pos))

        self.main.candidate_pnt_table.customContextMenuRequested.connect(lambda pos: showQueryContextMenu(self.main.candidate_pnt_table, pos))
        self.main.pending_pnt_table.customContextMenuRequested.connect(lambda pos: showQueryContextMenu(self.main.pending_pnt_table, pos))

        self.main.candidate_pnt_table.allToX.connect(lambda: movePoints(self.main.candidate_pnt_table, self.main.x_table))
        self.main.candidate_pnt_table.selectedToX.connect(lambda: movePoints(self.main.candidate_pnt_table, self.main.x_table, selected=True))
        self.main.candidate_pnt_table.allToOther.connect(lambda: movePoints(self.main.candidate_pnt_table, self.main.pending_pnt_table))
        self.main.candidate_pnt_table.selectedToOther.connect(lambda: movePoints(self.main.candidate_pnt_table, self.main.pending_pnt_table, selected=True))
        self.main.pending_pnt_table.allToX.connect(lambda: movePoints(self.main.pending_pnt_table, self.main.x_table))
        self.main.pending_pnt_table.selectedToX.connect(lambda: movePoints(self.main.pending_pnt_table, self.main.x_table, selected=True))
        self.main.pending_pnt_table.allToOther.connect(lambda: movePoints(self.main.pending_pnt_table, self.main.candidate_pnt_table))
        self.main.pending_pnt_table.selectedToOther.connect(lambda: movePoints(self.main.pending_pnt_table, self.main.candidate_pnt_table, selected=True))

        self.main.x_table.verticalScrollBar().valueChanged.connect(self.main.y_table.verticalScrollBar().setValue)
        self.main.x_table.horizontalScrollBar().valueChanged.connect(onHorizontalScrollBarChange)
        self.main.y_table.verticalScrollBar().valueChanged.connect(self.main.x_table.verticalScrollBar().setValue)
        self.main.candidate_pnt_table.horizontalScrollBar().valueChanged.connect(onHorizontalScrollBarChange)
        self.main.pending_pnt_table.horizontalScrollBar().valueChanged.connect(onHorizontalScrollBarChange)
        self.main.boundry_table.horizontalScrollBar().valueChanged.connect(onHorizontalScrollBarChange)

        self.main.epoch_label.textChanged.connect(lambda epoch: on_epoch_change(int(epoch)))

        self.main.ucb_button.clicked.connect(lambda: self.main.beta_spinbox.setEnabled(True))
        self.main.kg_button.clicked.connect(lambda: self.main.beta_spinbox.setEnabled(False))
        self.main.update_gp_button.clicked.connect(updateGP)
        self.main.query_candidates_button.clicked.connect(queryGP)
        self.main.obj_func_btn.clicked.connect(self.main.obj_func_win.show)
        self.main.obj_func_win.gotCustomFunction.connect(applyObjFuncToY)
        self.plots.acq_min_button.clicked.connect(lambda: self.plots.acq_fixed_table.setEnabled(False))
        self.plots.acq_mean_button.clicked.connect(lambda: self.plots.acq_fixed_table.setEnabled(False))
        self.plots.acq_fix_button.clicked.connect(lambda: self.plots.acq_fixed_table.setEnabled(True))
        self.plots.post_min_button.clicked.connect(lambda: self.plots.post_fixed_table.setEnabled(False))
        self.plots.post_mean_button.clicked.connect(lambda: self.plots.post_fixed_table.setEnabled(False))
        self.plots.post_fix_button.clicked.connect(lambda: self.plots.post_fixed_table.setEnabled(True))
        self.plots.update_proj_button.clicked.connect(lambda: self.GPBO.plot(main=True) if self.GPBO is not None else None)
        self.main.progress_button.progressBarEnabled.connect(on_progress_bar_enable)

        self.plots.acq_x_combobox.currentIndexChanged.connect(lambda: onPlotsComboBoxChange(self.plots.acq_x_combobox, self.plots.acq_y_combobox))
        self.plots.acq_y_combobox.currentIndexChanged.connect(lambda: onPlotsComboBoxChange(self.plots.acq_y_combobox, self.plots.acq_x_combobox))
        self.plots.post_x_combobox.currentIndexChanged.connect(lambda: onPlotsComboBoxChange(self.plots.post_x_combobox, self.plots.post_y_combobox))
        self.plots.post_y_combobox.currentIndexChanged.connect(lambda: onPlotsComboBoxChange(self.plots.post_y_combobox, self.plots.post_x_combobox)) 

        self.plots.epoch_spinbox.valueChanged.connect(onPlotsEpochChange)

        self.menu_bar.newFile.connect(on_new_file)
        self.menu_bar.openFile.connect(on_open_file)
        self.menu_bar.refreshPlots.connect(on_plot_refresh)
        self.menu_bar.saveAsFile.connect(on_save_as)
        self.menu_bar.preferences.connect(on_open_preferences)
        
        self.main.obj_func_win.done_btn.clicked.connect(applyObjFuncToY)

        self.menu_bar.pref_win.app_font_sb.valueChanged.connect(app_font_size_change)
        self.menu_bar.pref_win.path_log_le.editingFinished.connect(lambda: verify_path(self.menu_bar.pref_win.path_log_le))
        self.menu_bar.priorRequested.connect(self.menu_bar.prior_win.show)
        self.menu_bar.prior_win.gotCustomFunction.connect(got_prior)

        self.centralWidget().tabs.tabEnabled.connect(on_tab_enabled)



class SuperGPBO(interactiveGPBO):
    """
    This is a super class of the interactive GPBO. Its purpose is to add another
    level of abstraction on top of the interactive GPBO, which is subject to 
    changes and requires many function parameters.
    """
    def __init__(self, main_window: QWidget, filename: str = None, regular_init: bool = False,*args, **kwargs):
        """
        Initializes the GPBO. If a file name is supplied, it will load the file 
        instead of running its regular procedures.
        
        By supplying it the main window of BO-GUI, you are giving it access to 
        all QWidgets within the program. While this sacrifices reusability, it 
        improves on abstraction so greatly that I believe it is worth utilizing.

        Args:
            main_window (QWidget): The main window of BO-GUI.
            filename (str, optional): Name of file to load. Defaults to None.
        """
        self.control: Controller = main_window
        

        if filename:
            super().__init__(load_log_fname=filename)
            self.updateProgressBar.connect(self.control.main.progress_button.updateValue)
            self.updateProgressBar.connect(self.control.plots.progress_button.updateValue)
            return
        elif regular_init:
            super().__init__(*args, **kwargs)
            self.updateProgressBar.connect(self.control.main.progress_button.updateValue)
            self.updateProgressBar.connect(self.control.plots.progress_button.updateValue)
            return
        
        
        x, y = self._xy()
        
        bounds = self._bounds()
        batch_size = self._batch_size()
        acq_func, acq_args = self._acq()
        prior = self._prior()
        
        super().__init__(x, y, bounds=bounds, batch_size=batch_size,
                         acquisition_func=acq_func, 
                         acquisition_func_args=acq_args, 
                         prior_mean_model=prior)
        self.updateProgressBar.connect(self.control.main.progress_button.updateValue)
        self.updateProgressBar.connect(self.control.plots.progress_button.updateValue)
                
    def _bool_projection(self, projection_type: str) -> Tuple[bool, bool]:
        """Converts projection type into a boolean sequence.

        Args:
            projection_type (str): "Minimum" or "Mean".

        Returns:
            tuple[bool, bool]: Minimum and mean projection
            types.
        """
        minimum = False
        mean = False
        
        if projection_type == "Minimum":
            minimum = True
        elif projection_type == "Mean":
            mean = True
            
        return minimum, mean
    def _xy(self) -> Tuple[np.array, np.array]:
        """Gets X and Y. 

        Returns:
            tuple[np.array, np.array]: X and Y.
        """
        # Remove last row to account for table's auto-resizing
        x = self.control.main.x_table.to_list()
        y = self.control.main.y_table.to_list()
        
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

        return np.array(remaining_x, dtype=DTYPE), np.array(remaining_y, dtype=DTYPE)
    def _batch_size(self) -> int:
        """Retrieves the batch size.

        Returns:
            int: Batch size.
        """
        return self.control.main.batch_spin_box.value()
    def _acq(self) -> Tuple[str, Union[Dict[str, float], None]]:
        """Retrieves the acquisition function and its arguments.

        Returns:
            tuple[str, dict[str, float] | None]: Acquisition function and its 
            arguments
        """        
        if self.control.main.ucb_button.isChecked():
            acq_func = 'UCB'
            acq_func_args = {'beta': self.control.main.beta_spinbox.value()}
        elif self.control.main.kg_button.isChecked():
            acq_func = 'KG'
            acq_func_args = None
            
        return acq_func, acq_func_args
    def _bounds(self) -> np.array:
        """Retrieves the boundary.

        Returns:
            np.array: Boundary.
        """
        bounds = self.control.main.boundry_table.to_list()

        for element in bounds:
            if len(element) < 2:
                raise ValueError("Boundary incomplete.")
            elif element[0] > element[1]:
                raise ValueError("Boundary minimum must be less than its corresponding maximum.")

        return np.array(bounds, dtype=DTYPE)
    def _prior(self):
        # self._prior_mean = 
        pass
    def _epoch(self) -> int:
        """Retrieves the epoch.

        Returns:
            int: Epoch.
        """
        return len(self.history) - 1
    def _update_epoch(self):
        """
        Updates the on-screen epoch label.
        """
        epoch = self._epoch()
        self.control.main.epoch_label.setText(str(epoch))
    def _pending_pnts(self) -> Union[np.array, None]:
        """Retrieves pending points.

        Returns:
            np.array | None: Pending points or nothing.
        """
        pending_pnts = self.control.main.pending_pnt_table.to_list()
        
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
                raise ValueError("GPBO cancelled.")
            
        return remaining if remaining else None
    def _get_info(self, gp=False, proj=False) -> Tuple[Dict[str, Union[List[Axes], str, int, Union[Dict[int, np.float32], None], str]], Dict[str, Union[Axes, str, int, Union[Dict[int, np.float32], None], str]], Dict[str, Axes]]:
        """
        Retrieves necessary information and puts it into a neat dictionary.
        
        Args:
            gp (bool, optional): Do you need the preview axis? Defaults to False.
            proj (bool, optional): Do you need the main axis? Defaults to False.

        Returns:
            dict: Acquisition information.
            dict: Posterior information.
            dict: Objective History information.
            
        Keys:
            "Axes" (List[Axes]): Requested posterior axes.
            "X Label" (str): Current x-label.
            "Y Label" (str): Current y-label.
            "X Dimension" (int): Represents current x-axis dimension.
            "Y Dimension" (int): Represents current y-axis dimension.
            "Fixed Values" (Union[dict, None]): Current fixed dimensional values.
            "Projection Type" (str): "Minimum", "Mean", or "Fixed".
        """
        DECISION_HEADER = self.control.main.x_table.get_horizontal_labels()
        EPOCH = self._epoch()
        
        if not gp and not proj:
            raise ValueError("Must select `GP` or `proj`.")

        # ACQUISITION
        axes = []
        if gp:
            axes.append(self.control.main.canvas.acquisition_ax)
        if proj:
            axes.append(self.control.plots.canvas.acquisition_ax)
        
        acq = dict()
        acq["Axes"] = axes
        acq["X Label"] = self.control.plots.acq_x_combobox.currentText() if EPOCH > 1 else DECISION_HEADER[0]
        acq["Y Label"] = self.control.plots.acq_y_combobox.currentText() if EPOCH > 1 else DECISION_HEADER[1]
        acq["X Dimension"] = self.control.plots.acq_x_combobox.currentIndex() if EPOCH > 1 else DECISION_HEADER.index(acq["X Label"])
        acq["Y Dimension"] = self.control.plots.acq_y_combobox.currentIndex() if EPOCH > 1 else DECISION_HEADER.index(acq["Y Label"])
        acq["Fixed Values"] = None
        
        if EPOCH > 1:
            if self.control.plots.acq_min_button.isChecked():
                acq["Projection Type"] = "Minimum"
            elif self.control.plots.acq_mean_button.isChecked():
                acq["Projection Type"] = "Mean"
            else:
                acq["Projection Type"] = "Fixed"
                acq["Fixed Values"] = self.control.plots.acq_fixed_table.to_dim()
        else:
            acq["Projection Type"] = "Minimum"
            
        # POSTERIOR
        axes = []
        if gp:
            axes.append(self.control.main.canvas.posterior_ax)
        if proj:
            axes.append(self.control.plots.canvas.posterior_ax)
        
        post = dict()
        post["Axes"] = axes
        post["X Label"] = self.control.plots.post_x_combobox.currentText() if EPOCH > 1 else DECISION_HEADER[0]
        post["Y Label"] = self.control.plots.post_y_combobox.currentText() if EPOCH > 1 else DECISION_HEADER[1]
        post["X Dimension"] = self.control.plots.post_x_combobox.currentIndex() if EPOCH > 1 else DECISION_HEADER.index(acq["X Label"])
        post["Y Dimension"] = self.control.plots.post_y_combobox.currentIndex() if EPOCH > 1 else DECISION_HEADER.index(acq["Y Label"])
        post["Fixed Values"] = None

        if EPOCH > 1:
            if self.control.plots.post_min_button.isChecked():
                post["Projection Type"] = "Minimum"
            elif self.control.plots.post_mean_button.isChecked():
                post["Projection Type"] = "Mean"
            else:
                post["Projection Type"] = "Fixed"
                post["Fixed Values"] = self.control.plots.post_fixed_table.to_dim()
        else:
            post["Projection Type"] = "Minimum"
        
        # OBJECTIVE HISTORY
        axes = []
        if gp:
            axes.append(self.control.main.canvas.obj_history_ax)
        if proj:
            axes.append(self.control.plots.canvas.obj_history_ax)
        
        obj = dict()
        obj["Axes"] = axes
    
        return acq, post, obj
    def plot(self, preview=False, main=False) -> None:
        """
        The standard plot for BO-GUI canvases. Must pick one option or the other.

        Args:
            preview (bool, optional): Plot onto the preview canvas. Defaults to False.
            main (bool, optional): Plot onto the main canvas. Defaults to False.
        """
        if (preview and main) or (not preview and not main):
            raise ValueError("Must select `preview` or `main`.")
        
        def wrapper():
            print("Plotting...")
            
            if self.control.main.plot_button.isChecked() or main:
                print("!")
                self.control.main.progress_button.enableProgressBar()
                self.control.plots.progress_button.enableProgressBar()
                print("@")
                if preview:
                    epoch = self._epoch()
                    canvas = self.control.main.canvas
                else:
                    epoch = self.control.plots.epoch_spinbox.value()
                    canvas = self.control.plots.canvas
                print(epoch)
                    
                query_cb = self.control.plots.query_combobox
                i = None
                if main and query_cb.currentText() != "All":
                    i = int(query_cb.currentText()) - 1
                    
                acq, post, obj = self._get_info(gp=preview, proj=main)
                acq_min, acq_mean = self._bool_projection(acq["Projection Type"])
                post_min, post_mean = self._bool_projection(post["Projection Type"])

                acq_labels = acq["X Label"], acq["Y Label"]
                post_labels = post["X Label"], post["Y Label"]
                
                canvas.clear()
                print("*******")
                print(acq)
                try:
                    self.plot_LCB_2D_projection(self.control, 
                                                epoch=epoch, 
                                                axes=acq["Axes"],
                                                dim_xaxis=acq["X Dimension"], 
                                                dim_yaxis=acq["Y Dimension"],
                                                project_minimum=acq_min,
                                                project_mean=acq_mean,
                                                i_query=i,
                                                fixed_values_for_each_dim=acq["Fixed Values"])
                    self.plot_GPmean_2D_projection(self.control,
                                                epoch=epoch,
                                                axes=post["Axes"],
                                                dim_xaxis=post["X Dimension"],
                                                dim_yaxis=post["Y Dimension"],
                                                project_minimum=post_min,
                                                project_mean=post_mean,
                                                fixed_values_for_each_dim=post["Fixed Values"])
                except RuntimeError:
                    # QMessageBox.warning(self.control, "WARNING", "Plotting can take a long time... continue?", QMessageBox.Ok | QMessageBox.Cancel)
                    self.plot_LCB_2D_projection(self.control, 
                                                epoch=epoch, 
                                                axes=acq["Axes"],
                                                dim_xaxis=acq["X Dimension"], 
                                                dim_yaxis=acq["Y Dimension"],
                                                project_minimum=acq_min,
                                                project_mean=acq_mean,
                                                i_query=i,
                                                fixed_values_for_each_dim=acq["Fixed Values"],
                                                overdrive=True)
                    self.plot_GPmean_2D_projection(self.control,
                                                epoch=epoch,
                                                axes=post["Axes"],
                                                dim_xaxis=post["X Dimension"],
                                                dim_yaxis=post["Y Dimension"],
                                                project_minimum=post_min,
                                                project_mean=post_mean,
                                                fixed_values_for_each_dim=post["Fixed Values"],
                                                overdrive=True)
                    
                self.plot_obj_history(axes=obj["Axes"])
                
                if epoch == 1 and self.control.main.plot_button.isEnabled():
                    # Show selectcanvas axes
                    canvas.acquisition_ax.set_visible(True)
                    canvas.posterior_ax.set_visible(True)
                    canvas.obj_history_ax.set_visible(False)
                    canvas._get_obj_twinx().set_visible(False)
                elif self.control.main.plot_button.isEnabled() and epoch > 0:
                    # Show all canvas axes
                    for ax in canvas.get_axes():
                        if not ax.get_visible():
                            ax.set_visible(True)
                else:
                    # Hide select canvas axes
                    canvas.acquisition_ax.set_visible(False)
                    canvas.posterior_ax.set_visible(False)
                
                if main:
                    canvas.format(acq_labels, post_labels)
                    canvas.reload()  
                elif preview:
                    canvas.format(acq_labels, post_labels, 
                                  compact=True, legends=False)
                    canvas.reload(compact=True)
            elif self._epoch() > 0:
                if preview:
                    epoch = self._epoch()
                    canvas = self.control.main.canvas
                else:
                    epoch = self.control.plots.epoch_spinbox.value()
                    canvas = self.control.plots.canvas
                    
                acq, post, obj = self._get_info(gp=preview, proj=main)
                
                canvas.clear()
                for cbar in self.control.colorbars.values():
                    cbar.remove()
                self.control.colorbars.clear()
                
                self.plot_obj_history(axes=obj["Axes"])

                canvas.acquisition_ax.set_visible(False)
                canvas.posterior_ax.set_visible(False)
                canvas.obj_history_ax.set_visible(True)
                canvas._get_obj_twinx().set_visible(True)
                
                acq_labels = acq["X Label"], acq["Y Label"]
                post_labels = post["X Label"], post["Y Label"]
                canvas.format(acq_labels, post_labels, 
                                  compact=True, legends=False)
                canvas.reload(compact=True)
                
                
                

        thread = threading.Thread(target=wrapper)
        thread.start()
    def refresh_plots(self):
        print("Refreshing plots....")
        
        acq, post, _ = self._get_info(gp=True, proj=True)

        acq_labels = acq["X Label"], acq["Y Label"]
        post_labels = post["X Label"], post["Y Label"]
        
        self.control.plots.canvas.format(acq_labels, post_labels)
        self.control.plots.canvas.reload()
        
        self.control.main.canvas.format(acq_labels, post_labels, 
                                        compact=True, legends=False)
        self.control.main.canvas.reload(compact=True)   
    def update(self) -> None:
        """Update the GP."""
        print("Updating GP...")
        
        x, y = self._xy()
        
        self.update_GP(x=x,y=y)
        self._update_epoch()

        self.plot(preview=True)
    def query(self) -> None:
        """Query for candidate points."""
        def calcDistance(pnt_A, pnt_B, weights=None):
            distances = []
            for i in range(len(pnt_A)):
                coord_A = pnt_A[i]
                coord_B = pnt_B[i]
                if weights:
                    distances.append(abs(coord_A - coord_B) * weights[i])
                else:
                    distances.append(abs(coord_A - coord_B))
            return max(distances)
        def findBestPoint(prev_pnt, eval_pnts, weights=None):
            # calculating distances
            eval_pnt_distances = []
            for eval_pnt in eval_pnts:
                    distance = calcDistance(prev_pnt, eval_pnt, weights)
                    eval_pnt_distances.append(distance)
                    
            pnt_index = eval_pnt_distances.index(min(eval_pnt_distances))
            return eval_pnts[pnt_index], pnt_index
        def orderByDistance(prev_pnt, eval_pnts, weights=None):
            ordered_pnts = []

            if not weights or len(eval_pnts[0]) != len(weights):
                weights = []

            while eval_pnts.size != 0:
                best_pnt, best_pnt_index = findBestPoint(prev_pnt, eval_pnts, weights)
                ordered_pnts.append(best_pnt)
                eval_pnts = np.delete(eval_pnts, best_pnt_index, 0)
            return np.array(ordered_pnts)
        
        def wrapper():
            print("Querying candidates...")

            bsize = self._batch_size()
            epoch = self._epoch()
            pending_pnts = self._pending_pnts()
            
            candidates = self.query_candidates(batch_size=bsize, X_pending=pending_pnts)
            candidates = orderByDistance(self.x[-1], candidates, weights=None)
            print(f"\t{candidates.tolist()}")
            
            candidates = candidates.tolist()
            self.control.main.candidate_pnt_table.setRowCount(len(candidates))
            self.control.main.candidate_pnt_table.fill(candidates)
            
            if self.control.plots.epoch_spinbox.value() == epoch:
                candidate_pnts = self.history[epoch]['x1']

                for i in range(1, self.control.plots.query_combobox.count()):
                    self.control.plots.query_combobox.removeItem(1)
                
                for i in range(len(candidate_pnts) - 1):
                    self.control.plots.query_combobox.addItem(f'{i+1}')
            
            if self.control.main.plot_button.isChecked():
                self.plot(preview=True)
            self.control.main.query_candidates_button.setEnabled(True)
            
        thread = threading.Thread(target=wrapper)
        thread.start()