from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('BO-GUI')
        self.setCentralWidget(QWidget())
        central_layout = QHBoxLayout()
        self.centralWidget().setLayout(central_layout)

        tabs = QTabWidget()
        tabs.addTab(self.createInitializationPage(), 'Initialization')
        tabs.addTab(self.createIterationPage(), 'Iteration')

        central_layout.addWidget(tabs)
        

    def createInitializationPage(self):
        from classes.tables import DecisionParameterTable, ObjectiveTable, BoundryTable
        from classes.utility import AcquisitionWidget
        
        # data
        self.initial_x_table = DecisionParameterTable(3, 4)
        self.initial_y_table = ObjectiveTable(3)
        self.bounds_table = BoundryTable(4)
        
        data_layout = QGridLayout()
        data_layout.addWidget(QLabel('Initial Decision Parameters'), 0, 0, 1, 3, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.initial_x_table, 1, 0, 1, 3)
        data_layout.addWidget(QLabel('Initial Objective'), 0, 3, 1, 1, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.initial_y_table, 1, 3, 1, 1)
        data_layout.addWidget(QLabel('Boundary of Decision'), 2, 0, 1, 3, alignment=Qt.AlignCenter)
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
        self.aux_arguments_line_edit = QLineEdit()
        self.log_data_path_line_edit = QLineEdit()
        self.prior_mean_model_path_line_edit = QLineEdit()
        
        opt_layout = QGridLayout()
        opt_layout.addWidget(QLabel('Regularization Coefficient:'), 0, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.reg_coeff_spinbox, 0, 1, 1, 1)
        opt_layout.addWidget(QLabel('Avoid Bound Corners:'), 1, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.bounds_corners_check_box, 1, 1, 1, 1)
        opt_layout.addWidget(QLabel('Auxiliary Arguments:'), 2, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.aux_arguments_line_edit, 2, 1, 1, 1)
        opt_layout.addWidget(QLabel('Log Data Path:'), 3, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.log_data_path_line_edit, 3, 1, 1, 1)
        opt_layout.addWidget(QLabel('Prior Mean Model Path:'), 4, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.prior_mean_model_path_line_edit, 4, 1, 1, 1)

        opt_groupbox = QGroupBox('Optional')
        opt_groupbox.setLayout(opt_layout)

        # main
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(acq_groupbox, 1)
        right_layout.addWidget(opt_groupbox, 3)

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
        from matplotlib.backends.backend_qtagg import FigureCanvas
        from classes.utility import AcquisitionWidget, IterateButton
        
        # display
        self.epoch_label = QLabel('0')
        self.num_obj_data_label = QLabel('0')
        
        display_layout = QGridLayout()
        display_layout.addWidget(QLabel('Iteration Epoch:'), 0, 0, 1, 1, alignment=Qt.AlignRight)
        display_layout.addWidget(self.iteration_epoch_label, 0, 1, 1, 1)
        display_layout.addWidget(QLabel('Number of Objective Evaluation Data:'), 1, 0, 1, 1, alignment=Qt.AlignRight)
        display_layout.addWidget(self.num_obj_data_label, 1, 1, 1, 1)
        
        display_groupbox = QGroupBox('Display')
        display_groupbox.setLayout(display_layout)

        # plots
        self.canvas = FigureCanvas()

        plots_layout = QVBoxLayout()
        plots_layout.addWidget(self.canvas)
        
        plots_groupbox = QGroupBox('Plots')
        plots_groupbox.setLayout(plots_layout)

        # control
        self.batch_spin_box = QSpinBox()
        batch_layout = QHBoxLayout()
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.addWidget(QLabel('Batch Size:'))
        batch_layout.addWidget(self.batch_spin_box)

        batch_widget = QWidget()
        batch_widget.setLayout(batch_layout)
        
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        
        self.iteration_acq_widget = AcquisitionWidget()
        
        control_layout = QGridLayout()
        control_layout.addWidget(batch_widget, 0, 0, 1, 1)
        control_layout.addWidget(h_line, 1, 0, 1, 6)
        control_layout.addWidget(self.iteration_acq_widget, 2, 0, 1, 6)

        control_groupbox = QGroupBox('Control')
        control_groupbox.setLayout(control_layout)
        
        # data 
        self.iteration_x_table = DecisionParameterTable(3, 4)
        self.iteration_y_table = ObjectiveTable(3)

        data_layout = QGridLayout()
        data_layout.addWidget(QLabel('Decision Parameters'), 0, 0, 1, 1, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.iteration_x_table, 1, 0, 1, 1)
        data_layout.addWidget(QLabel('Objective'), 0, 1, 1, 1, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.iteration_y_table, 1, 1, 1, 1)
        
        data_groupbox = QGroupBox('Data')
        data_groupbox.setLayout(data_layout)

        # main
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(display_groupbox,1)
        left_layout.addWidget(plots_groupbox, 5)
        right_layout.addWidget(control_groupbox, 1)
        right_layout.addWidget(data_groupbox, 5)
        right_layout.addWidget(IterateButton(), 1)

        left_qwidget = QWidget()
        right_qwidget = QWidget()
        left_qwidget.setLayout(left_layout)
        right_qwidget.setLayout(right_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_qwidget, 1)
        main_layout.addWidget(right_qwidget, 3)
        
        main_qwidget = QWidget()
        main_qwidget.setLayout(main_layout)
        return main_qwidget