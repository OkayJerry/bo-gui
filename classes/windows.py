from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import os


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        from classes.utility import MenuBar, ProgressDialog  # , ConvenienceTabWidget
        import globals as glb

        super().__init__(parent=parent)
        self.setWindowTitle('BO-GUI')
        self.setWindowIcon(QIcon('images/frib.png'))
        self.setCentralWidget(QWidget())
        self.setMenuBar(MenuBar(self))

        self.progress_dialog = ProgressDialog(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.createInitializationPage(), 'Initialization')
        self.tabs.addTab(self.createIterationPage(), 'Iteration')
        self.tabs.addTab(self.createPlotsPage(), 'Plots')
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)

        central_layout = QHBoxLayout()
        central_layout.addWidget(self.tabs)
        self.centralWidget().setLayout(central_layout)

    def createInitializationPage(self):
        from classes.tables import DecisionParameterTable, ObjectiveTable, BoundryTable
        from classes.utility import AcquisitionWidget, InitializeButton

        # data
        self.initial_x_table = DecisionParameterTable(1, 1)
        self.initial_y_table = ObjectiveTable(1)
        self.bounds_table = BoundryTable(1)

        self.initial_x_table.itemChanged.connect(
            lambda: self.handleTableRowSynchronization(self.initial_x_table))
        self.initial_y_table.itemChanged.connect(
            lambda:  self.handleTableRowSynchronization(self.initial_y_table))
        self.initial_x_table.verticalScrollBar().valueChanged.connect(
            lambda: self.initial_x_table.syncScroll(self.initial_y_table))
        self.initial_y_table.verticalScrollBar().valueChanged.connect(
            lambda: self.initial_y_table.syncScroll(self.initial_x_table))

        data_layout = QGridLayout()
        data_layout.addWidget(
            QLabel('Initial Decision Parameters'), 0, 0, 1, 3, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.initial_x_table, 1, 0, 1, 3)
        data_layout.addWidget(QLabel('Initial Objective'),
                              0, 3, 1, 1, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.initial_y_table, 1, 3, 1, 1)
        data_layout.addWidget(QLabel('Boundary of Decision'),
                              2, 0, 1, 3, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.bounds_table, 3, 0, 1, 3)

        data_groupbox = QGroupBox('Data')
        data_groupbox.setLayout(data_layout)

        # acquistion
        self.initial_acq_widget = AcquisitionWidget()

        acq_layout = QVBoxLayout()
        acq_layout.addWidget(self.initial_acq_widget)

        acq_groupbox = QGroupBox('Acquisition')
        acq_groupbox.setLayout(acq_layout)

        # optional
        self.reg_coeff_spinbox = QSpinBox()
        self.bounds_corners_check_box = QCheckBox()
        self.log_data_path_line_edit = QLineEdit()
        self.prior_mean_model_path_line_edit = QLineEdit()
        self.aux_arguments_line_edit = QLineEdit()

        self.log_data_path_line_edit.setText(os.getcwd())

        opt_layout = QGridLayout()
        opt_layout.addWidget(
            QLabel('Regularization Coefficient:'), 0, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.reg_coeff_spinbox, 0, 1, 1, 1)
        opt_layout.addWidget(QLabel('Avoid Bound Corners:'),
                             1, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.bounds_corners_check_box, 1, 1, 1, 1)
        opt_layout.addWidget(QLabel('Log Data Path:'), 2,
                             0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.log_data_path_line_edit, 2, 1, 1, 1)
        opt_layout.addWidget(QLabel('Prior Mean Model Path:'),
                             3, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.prior_mean_model_path_line_edit, 3, 1, 1, 1)
        opt_layout.addWidget(
            QLabel('Prior Mean Auxiliary Arguments:'), 4, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.aux_arguments_line_edit, 4, 1, 1, 1)

        opt_groupbox = QGroupBox('Optional')
        opt_groupbox.setLayout(opt_layout)

        # main
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(acq_groupbox, 1)
        right_layout.addWidget(opt_groupbox, 3)
        right_layout.addWidget(InitializeButton())

        right_qwidget = QWidget()
        right_qwidget.setLayout(right_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(data_groupbox, 3)
        main_layout.addWidget(right_qwidget, 1)

        main_qwidget = QWidget()
        main_qwidget.setLayout(main_layout)
        return main_qwidget

    def createIterationPage(self):
        from classes.tables import DecisionParameterTable, ObjectiveTable
        from classes.canvas import PreviewCanvas
        from classes.utility import EvaluationPointGroupBox, IterateButton

        # display
        self.epoch_label = QLabel('0')
        self.num_obj_data_label = QLabel('0')

        display_layout = QGridLayout()
        display_layout.addWidget(
            QLabel('Iteration Epoch:'), 0, 0, 1, 1, alignment=Qt.AlignRight)
        display_layout.addWidget(self.epoch_label, 0, 1, 1, 1)
        display_layout.addWidget(QLabel(
            'Number of Objective Evaluation Data:'), 1, 0, 1, 1, alignment=Qt.AlignRight)
        display_layout.addWidget(self.num_obj_data_label, 1, 1, 1, 1)

        display_groupbox = QGroupBox('Display')
        display_groupbox.setLayout(display_layout)

        # plots
        self.preview_canvas = PreviewCanvas()

        plots_layout = QVBoxLayout()
        plots_layout.addWidget(self.preview_canvas)

        plots_groupbox = QGroupBox('Preview Plots')
        plots_groupbox.setLayout(plots_layout)

        # control
        self.iteration_batch_spin_box = QSpinBox()
        self.iteration_batch_spin_box.setRange(1, 100)
        self.iteration_beta_spin_box = QDoubleSpinBox()

        control_layout = QGridLayout()
        control_layout.addWidget(
            QLabel('Batch Size:'), 0, 0, 1, 1, alignment=Qt.AlignRight)
        control_layout.addWidget(self.iteration_batch_spin_box, 0, 1, 1, 2)
        control_layout.addWidget(
            QLabel('Beta:'), 1, 0, 1, 1, alignment=Qt.AlignRight)
        control_layout.addWidget(self.iteration_beta_spin_box, 1, 1, 1, 2)

        control_groupbox = QGroupBox('Control')
        control_groupbox.setLayout(control_layout)

        # data
        self.iteration_x_table = DecisionParameterTable(1, 1)
        self.iteration_y_table = ObjectiveTable(1)

        self.iteration_x_table.verticalScrollBar().valueChanged.connect(
            lambda: self.iteration_x_table.syncScroll(self.iteration_y_table))
        self.iteration_y_table.verticalScrollBar().valueChanged.connect(
            lambda: self.iteration_y_table.syncScroll(self.iteration_x_table))

        data_layout = QGridLayout()
        data_layout.addWidget(QLabel('Decision Parameters'),
                              0, 0, 1, 1, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.iteration_x_table, 1, 0, 1, 1)
        data_layout.addWidget(QLabel('Objective'), 0, 1,
                              1, 1, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.iteration_y_table, 1, 1, 1, 1)

        data_groupbox = QGroupBox('Data')
        data_groupbox.setLayout(data_layout)

        # top-right
        self.evaluation_point_groupbox = EvaluationPointGroupBox()

        top_right_layout = QHBoxLayout()
        top_right_layout.setContentsMargins(0, 0, 0, 0)
        top_right_layout.addWidget(control_groupbox, 1)
        top_right_layout.addWidget(self.evaluation_point_groupbox, 3)

        top_right_qwidget = QWidget()
        top_right_qwidget.setLayout(top_right_layout)

        # main
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(display_groupbox, 1)
        left_layout.addWidget(plots_groupbox, 5)
        right_layout.addWidget(top_right_qwidget, 1)
        right_layout.addWidget(data_groupbox, 5)
        right_layout.addWidget(IterateButton(), 1)

        left_qwidget = QWidget()
        right_qwidget = QWidget()
        left_qwidget.setLayout(left_layout)
        right_qwidget.setLayout(right_layout)

        # splitter = QSplitter()
        # splitter.addWidget(left_qwidget)
        # splitter.addWidget(right_qwidget)
        main_layout = QHBoxLayout()
        # main_layout.addWidget(splitter)
        main_layout.addWidget(left_qwidget, 1)
        main_layout.addWidget(right_qwidget, 3)

        main_qwidget = QWidget()
        main_qwidget.setLayout(main_layout)
        return main_qwidget

    def createPlotsPage(self):
        from classes.canvas import Canvas
        from classes.utility import AcqPostButton, AxisComboBox, AcqPostFixUpdateButton
        from classes.tables import AcqPostFixTable

        self.canvas = Canvas()

        # acquisition control
        self.acq_min_button = AcqPostButton('Project Minimum')  # default
        self.acq_mean_button = AcqPostButton('Project Mean')
        self.acq_fix_button = AcqPostButton('Project By Fixing')
        self.acq_x_combobox = AxisComboBox()
        self.acq_y_combobox = AxisComboBox()
        self.acq_fixed_table = AcqPostFixTable(0)
        self.acq_fix_update_button = AcqPostFixUpdateButton(
            'Update Fixed Dimensions')

        acq_xy_layout = QGridLayout()
        acq_xy_layout.addWidget(QLabel('x-axis:'), 0, 0,
                                1, 1, alignment=Qt.AlignRight)
        acq_xy_layout.addWidget(self.acq_x_combobox, 0, 1, 1, 6)
        acq_xy_layout.addWidget(QLabel('y-axis:'), 1, 0,
                                1, 1, alignment=Qt.AlignRight)
        acq_xy_layout.addWidget(self.acq_y_combobox, 1, 1, 1, 6)
        acq_xy_qwidget = QWidget()
        acq_xy_qwidget.setLayout(acq_xy_layout)

        acq_layout = QVBoxLayout()
        acq_layout.addWidget(self.acq_min_button)
        acq_layout.addWidget(self.acq_mean_button)
        acq_layout.addWidget(self.acq_fix_button)
        acq_layout.addWidget(acq_xy_qwidget)
        acq_layout.addWidget(self.acq_fixed_table)
        acq_layout.addWidget(self.acq_fix_update_button)

        acq_groupbox = QGroupBox('Acquisition')
        acq_groupbox.setLayout(acq_layout)

        # posterior mean control
        self.post_min_button = AcqPostButton('Project Minimum')  # default
        self.post_mean_button = AcqPostButton('Project Mean')
        self.post_fix_button = AcqPostButton('Project By Fixing')
        self.post_x_combobox = AxisComboBox()
        self.post_y_combobox = AxisComboBox()
        self.post_fixed_table = AcqPostFixTable(0)
        self.post_fix_update_button = AcqPostFixUpdateButton(
            'Update Fixed Dimensions')

        post_xy_layout = QGridLayout()
        post_xy_layout.addWidget(QLabel('x-axis:'), 0,
                                 0, 1, 1, alignment=Qt.AlignRight)
        post_xy_layout.addWidget(self.post_x_combobox, 0, 1, 1, 6)
        post_xy_layout.addWidget(QLabel('y-axis:'), 1,
                                 0, 1, 1, alignment=Qt.AlignRight)
        post_xy_layout.addWidget(self.post_y_combobox, 1, 1, 1, 6)
        post_xy_qwidget = QWidget()
        post_xy_qwidget.setLayout(post_xy_layout)

        post_layout = QVBoxLayout()
        post_layout.addWidget(self.post_min_button)
        post_layout.addWidget(self.post_mean_button)
        post_layout.addWidget(self.post_fix_button)
        post_layout.addWidget(post_xy_qwidget)
        post_layout.addWidget(self.post_fixed_table)
        post_layout.addWidget(self.post_fix_update_button)

        post_groupbox = QGroupBox('Posterior Mean')
        post_groupbox.setLayout(post_layout)

        # main
        right_layout = QVBoxLayout()
        right_layout.addWidget(acq_groupbox)
        right_layout.addWidget(post_groupbox)

        right_qwidget = QWidget()
        right_qwidget.setLayout(right_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.canvas, 3)
        main_layout.addWidget(right_qwidget, 1)

        main_qwidget = QWidget()
        main_qwidget.setLayout(main_layout)
        return main_qwidget

    def transferDataFromInitialization(self):
        import globals as glb
        from classes.tables import TableItem, DoubleTableItem

        # table -> table
        self.iteration_x_table.setFromTable(self.initial_x_table)
        self.iteration_y_table.setFromTable(self.initial_y_table)
        self.evaluation_point_groupbox.evaluation_point_table.setFromTable(
            self.initial_x_table)

        # acquisition -> acquisition
        if self.initial_acq_widget.ucb_button.isChecked():
            self.iteration_beta_spin_box.setEnabled(True)
        else:
            self.iteration_beta_spin_box.setEnabled(False)

        beta_val = self.initial_acq_widget.ucb_spinbox.value()
        self.iteration_beta_spin_box.setValue(beta_val)

        self.initial_y_table.updateNumEvaluationData()

        # table -> plot controls
        comboboxes = [self.acq_x_combobox, self.acq_y_combobox,
                      self.post_x_combobox, self.post_y_combobox]
        for combobox in comboboxes:
            combobox.blockSignals(True)

        self.acq_min_button.setChecked(True)
        self.post_min_button.setChecked(True)

        initial_x_table = glb.main_window.initial_x_table
        for table in [self.acq_fixed_table, self.post_fixed_table]:

            table.setRowCount(0)
            for i in range(initial_x_table.columnCount()):
                header = initial_x_table.horizontalHeaderItem(i).text()

                # table
                table.insertRow(table.rowCount())
                table.setItem(table.rowCount() - 1, 0, DoubleTableItem(0.0))
                table.setVerticalHeaderItem(i, TableItem(header))

                # combobox
                if table is self.acq_fixed_table:
                    self.acq_x_combobox.addItem(header)
                    self.acq_y_combobox.addItem(header)
                else:
                    self.post_x_combobox.addItem(header)
                    self.post_y_combobox.addItem(header)

        self.acq_fixed_table.setEnabled(False)
        self.post_fixed_table.setEnabled(False)

        self.acq_x_combobox.setCurrentIndex(0)
        self.acq_y_combobox.setCurrentIndex(1)
        self.acq_fixed_table.hideRow(0)
        self.acq_fixed_table.hideRow(1)
        self.post_x_combobox.setCurrentIndex(0)
        self.post_y_combobox.setCurrentIndex(1)
        self.post_fixed_table.hideRow(0)
        self.post_fixed_table.hideRow(1)

        for combobox in comboboxes:
            combobox.blockSignals(False)

    def handleTableRowSynchronization(self, table):
        from classes.tables import DoubleTableItem

        if table == self.initial_x_table or table == self.initial_y_table:
            if self.initial_x_table.rowCount() > self.initial_y_table.rowCount():
                self.initial_y_table.setRowCount(
                    self.initial_x_table.rowCount())

                for i in range(self.initial_y_table.columnCount()):
                    self.initial_y_table.setItem(
                        self.initial_y_table.rowCount() - 1, i, DoubleTableItem(''))

            elif self.initial_x_table.rowCount() < self.initial_y_table.rowCount():
                self.initial_x_table.setRowCount(
                    self.initial_y_table.rowCount())

                for i in range(self.initial_x_table.columnCount()):
                    self.initial_x_table.setItem(
                        self.initial_x_table.rowCount() - 1, i, DoubleTableItem(''))

        elif table == self.iteration_x_table or table == self.iteration_y_table:
            if self.iteration_x_table.rowCount() > self.iteration_y_table.rowCount():
                self.iteration_y_table.setRowCount(
                    self.iteration_x_table.rowCount())

                for i in range(self.iteration_y_table.columnCount()):
                    self.iteration_y_table.setItem(
                        self.iteration_y_table.rowCount() - 1, i, DoubleTableItem(''))

            elif self.iteration_x_table.rowCount() < self.iteration_y_table.rowCount():
                self.iteration_x_table.setRowCount(
                    self.iteration_y_table.rowCount())

                for i in range(self.iteration_x_table.columnCount()):
                    self.iteration_x_table.setItem(
                        self.iteration_x_table.rowCount() - 1, i, DoubleTableItem(''))

    def returnToInitializiation(self):
        # clearing tables
        self.initial_x_table.reset()
        self.initial_y_table.reset()
        self.bounds_table.reset()

        # resetting controls
        self.initial_acq_widget.ucb_spinbox.setValue(0.00)
        self.initial_acq_widget.ucb_button.setChecked(True)
        self.initial_acq_widget.kg_button.setChecked(False)
        self.reg_coeff_spinbox.setValue(0)
        self.bounds_corners_check_box.setChecked(False)
        self.log_data_path_line_edit.setText(os.getcwd())
        self.prior_mean_model_path_line_edit.setText('')
        self.aux_arguments_line_edit.setText('')

        # tabs
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setCurrentIndex(0)
