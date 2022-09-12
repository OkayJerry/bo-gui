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
        
        # data
        data_layout = QGridLayout()
        data_layout.addWidget(QLabel('Initial Decision Parameters'), 0, 0, 1, 3, alignment=Qt.AlignCenter)
        data_layout.addWidget(DecisionParameterTable(3, 4), 1, 0, 1, 3)
        data_layout.addWidget(QLabel('Initial Objective'), 0, 3, 1, 1, alignment=Qt.AlignCenter)
        data_layout.addWidget(ObjectiveTable(3), 1, 3, 1, 1)
        data_layout.addWidget(QLabel('Boundary of Decision'), 2, 0, 1, 3, alignment=Qt.AlignCenter)
        data_layout.addWidget(BoundryTable(4), 3, 0, 1, 3)
        
        data_groupbox = QGroupBox('Data')
        data_groupbox.setLayout(data_layout)

        # acquistion
        acq_layout = QGridLayout()
        acq_layout.addWidget(QRadioButton('Upper Confidence Bound (UCB)'), 0, 0, 1, 6)
        acq_layout.addWidget(QLabel('Beta:'), 1, 1, 1, 1, alignment=Qt.AlignRight)
        acq_layout.addWidget(QDoubleSpinBox(), 1, 2, 1, 2)
        acq_layout.addWidget(QRadioButton('Knowledge Gradient (KG)'), 2, 0, 1, 6)

        acq_groupbox = QGroupBox('Acquisition')
        acq_groupbox.setLayout(acq_layout)
        
        # optional
        opt_layout = QGridLayout()
        opt_layout.addWidget(QLabel('Regularization Coefficient:'), 0, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(QSpinBox(), 0, 1, 1, 1)
        opt_layout.addWidget(QLabel('Avoid Bound Corners:'), 1, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(QCheckBox(), 1, 1, 1, 1)
        opt_layout.addWidget(QLabel('Auxiliary Arguments:'), 2, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(QLineEdit(), 2, 1, 1, 1)
        opt_layout.addWidget(QLabel('Log Data Path:'), 3, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(QLineEdit(), 3, 1, 1, 1)
        opt_layout.addWidget(QLabel('Prior Mean Model Path:'), 4, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(QLineEdit(), 4, 1, 1, 1)

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
        
        # display
        display_layout = QGridLayout()
        display_layout.addWidget(QLabel('Iteration Epoch:'), 0, 0, 1, 1, alignment=Qt.AlignRight)
        display_layout.addWidget(QLabel('999'), 0, 1, 1, 1)
        display_layout.addWidget(QLabel('Number of Objective Evaluation Data:'), 1, 0, 1, 1, alignment=Qt.AlignRight)
        display_layout.addWidget(QLabel('999'), 1, 1, 1, 1)
        
        display_groupbox = QGroupBox('Display')
        display_groupbox.setLayout(display_layout)

        # plots
        plots_layout = QVBoxLayout()
        plots_layout.addWidget(FigureCanvas())
        
        plots_groupbox = QGroupBox('Plots')
        plots_groupbox.setLayout(plots_layout)

        # control
        batch_layout = QHBoxLayout()
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.addWidget(QLabel('Batch Size:'))
        batch_layout.addWidget(QSpinBox())

        batch_widget = QWidget()
        batch_widget.setLayout(batch_layout)
        
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        
        control_layout = QGridLayout()
        control_layout.addWidget(batch_widget, 0, 0, 1, 1)
        control_layout.addWidget(h_line, 1, 0, 1, 6)
        control_layout.addWidget(QRadioButton('Upper Confidence Bound (UCB)'), 2, 0, 1, 6)
        control_layout.addWidget(QLabel('Beta:'), 3, 0, 1, 1, alignment=Qt.AlignRight)
        control_layout.addWidget(QDoubleSpinBox(), 3, 1, 1, 1)
        control_layout.addWidget(QRadioButton('Knowledge Gradient (KG)'), 4, 0, 1, 6)

        control_groupbox = QGroupBox('Control')
        control_groupbox.setLayout(control_layout)
        
        # data 
        data_layout = QHBoxLayout()
        data_layout.addWidget(DecisionParameterTable(3, 4), 3)
        data_layout.addWidget(ObjectiveTable(3), 1)
        
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
        right_layout.addWidget(QPushButton('Step Iteration'), 1)

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