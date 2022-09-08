from ctypes import alignment
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        from classes.tables import Table
        
        super().__init__(parent=parent)
        self.setWindowTitle('BO-GUI')
        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(QGridLayout())

        # data
        data_groupbox = QGroupBox('Data')
        data_groupbox.setLayout(QGridLayout())

        data_groupbox.layout().addWidget(Table(3, 4), 0, 0, 6, 1)
        data_groupbox.layout().addWidget(Table(3, 4), 0, 1, 4, 1)
        data_groupbox.layout().addWidget(Table(3, 4), 4, 1, 2, 1)
        
        # acquistion
        acq_groupbox = QGroupBox('Acquisition')
        acq_groupbox.setLayout(QGridLayout())

        acq_groupbox.layout().addWidget(QRadioButton('Upper Confidence Bound (UCB)'), 0, 0, 1, 6)
        acq_groupbox.layout().addWidget(QLabel('Beta:'), 1, 1, 1, 1, alignment=Qt.AlignRight)
        acq_groupbox.layout().addWidget(QDoubleSpinBox(), 1, 2, 1, 2)
        acq_groupbox.layout().addWidget(QRadioButton('Knowledge Gradient (KG)'), 2, 0, 1, 6)
        
        # optional
        opt_groupbox = QGroupBox('Optional')
        opt_groupbox.setLayout(QGridLayout())

        opt_groupbox.layout().addWidget(QLabel('Regularization Coefficient:'), 0, 0, 1, 1, alignment=Qt.AlignRight)
        opt_groupbox.layout().addWidget(QSpinBox(), 0, 1, 1, 1)
        opt_groupbox.layout().addWidget(QLabel('Avoid Bound Corners:'), 1, 0, 1, 1, alignment=Qt.AlignRight)
        opt_groupbox.layout().addWidget(QCheckBox(), 1, 1, 1, 1)
        opt_groupbox.layout().addWidget(QLabel('Auxiliary Arguments:'), 2, 0, 1, 1, alignment=Qt.AlignRight)
        opt_groupbox.layout().addWidget(QLineEdit(), 2, 1, 1, 1)
        opt_groupbox.layout().addWidget(QLabel('Log Data Path:'), 3, 0, 1, 1, alignment=Qt.AlignRight)
        opt_groupbox.layout().addWidget(QLineEdit(), 3, 1, 1, 1)
        opt_groupbox.layout().addWidget(QLabel('Prior Mean Model Path:'), 4, 0, 1, 1, alignment=Qt.AlignRight)
        opt_groupbox.layout().addWidget(QLineEdit(), 4, 1, 1, 1)

        # central widget
        central_layout = self.centralWidget().layout()
        central_layout.addWidget(data_groupbox, 0, 0, 4, 1)
        central_layout.addWidget(acq_groupbox, 0, 1, 1, 1)
        central_layout.addWidget(opt_groupbox, 1, 1, 3, 1)
        
        