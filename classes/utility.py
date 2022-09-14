from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np

class HeaderContextMenu(QMenu):
    def __init__(self, index, parent=None):
        super().__init__(parent)

        rename_action = QAction('Rename', self)
        rename_action.triggered.connect(lambda: self.rename(index))
        self.addAction(rename_action)
        
    def rename(self, index):
        import globals as glb
        
        parent_table = self.parent()
        old_header = parent_table.horizontalHeaderItem(index).text()
        new_header, ok = QInputDialog.getText(self, 'Rename Header', f'Header {index}:', QLineEdit.Normal, old_header)

        if ok:
            if parent_table in glb.tables['x0']:
                for table in glb.tables['x0']:
                    table.horizontalHeaderItem(index).setText(new_header)
            else:
                for table in glb.tables['y0']:
                    table.horizontalHeaderItem(index).setText(new_header)
            
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
            for acq_widget in glb.acq_widgets:
                acq_widget.ucb_button.blockSignals(True)

                acq_widget.ucb_button.setChecked(True)
                acq_widget.ucb_spinbox.setEnabled(True)

                acq_widget.ucb_button.blockSignals(False)
        else:
            for acq_widget in glb.acq_widgets:
                acq_widget.ucb_button.blockSignals(True)

                acq_widget.kg_button.setChecked(True)
                acq_widget.ucb_spinbox.setEnabled(False)

                acq_widget.ucb_button.blockSignals(False)
                
class IterateButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setText('Step Iteration')
        self.clicked.connect(self.onClick)

    def onClick(self):
        from classes.interactiveGPBO import InteractiveGPBO
        import globals as glb
        
        bounds = glb.main_window.bounds_table.getBounds()
        epoch_label = glb.main_window.epoch_label
        reg_coeff = glb.main_window.reg_coeff_spinbox.value()
        batch_size = glb.main_window.batch_spin_box.value()

        if int(epoch_label.text()) == 0:
            x0 = glb.main_window.initial_x_table.getArray()
            y0 = glb.main_window.initial_y_table.getArray()
            ucb_box = glb.main_window.initial_acq_widget.ucb_button
            
            if ucb_box.isChecked():
                beta_value = np.float64(glb.main_window.initial_acq_widget.ucb_spinbox.value())
                acq_func = 'UCB'
                acq_args = {'beta': beta_value}
            else:
                acq_func = 'KG'
                acq_args = {}           
            glb.interactive_GPBO = InteractiveGPBO(x0,y0,bounds=bounds, batch_size=batch_size, acquisition_func=acq_func, acquisition_func_args=acq_args, L2reg=reg_coeff)
        else:
            print('epoch_num > 0')
        #     glb.interactive_GPBO

        epoch_label.setText(str(int(epoch_label.text()) + 1))