from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class Table(QTableWidget):
    def __init__(self, rows, columns, parent=None):
        super().__init__(rows, columns, parent=parent)
        self.setAlternatingRowColors(True)
        
        horizontal_header = self.horizontalHeader()
        vertical_header = self.verticalHeader()
        horizontal_header.setSectionResizeMode(QHeaderView.Stretch)
        
        for header in [horizontal_header, vertical_header]:
            header.setContextMenuPolicy(Qt.CustomContextMenu)
            header.customContextMenuRequested.connect(self.customHeaderMenuRequested)
            
    def customHeaderMenuRequested(self, pos):
        from classes.utility import HeaderContextMenu
        
        if self.horizontalHeader().underMouse():
            header = self.horizontalHeader()
        else:
            header = self.verticalHeader()

        menu = HeaderContextMenu(header)
        menu.popup(header.viewport().mapToGlobal(pos))

class DecisionParameterTable(Table):
    def __init__(self, rows, columns, parent=None):
        super().__init__(rows, columns, parent)
        self.setHorizontalHeaderLabels([f'x0[{i}]' for i in range(columns)])
            
        
class ObjectiveTable(Table):
    def __init__(self, rows, parent=None):
        super().__init__(rows, 1, parent)
        self.setMinimumWidth(150)
        self.setMaximumWidth(150)
        self.setHorizontalHeaderLabels(['y0'])
        
class BoundryTable(Table):
    def __init__(self, columns, parent=None):
        super().__init__(2, columns, parent)
        self.setMaximumHeight(85)
        self.setMinimumHeight(85)
        self.setHorizontalHeaderLabels([f'x0[{i}]' for i in range(columns)])
        self.setVerticalHeaderLabels(['min', 'max'])