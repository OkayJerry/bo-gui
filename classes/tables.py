from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np

class Table(QTableWidget):
    def __init__(self, rows, columns, parent=None):
        super().__init__(rows, columns, parent=parent)
        self.setAlternatingRowColors(True)
        
        horizontal_header = self.horizontalHeader()
        horizontal_header.setSectionResizeMode(QHeaderView.Stretch)
        
        horizontal_header.setContextMenuPolicy(Qt.CustomContextMenu)
        horizontal_header.customContextMenuRequested.connect(self.customHeaderMenuRequested)

        for row in range(rows):
            for column in range(columns):
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignCenter)
                item.setText('0')
                self.setItem(row, column, item)
            
    def customHeaderMenuRequested(self, pos):
        from classes.utility import HeaderContextMenu
        
        menu = HeaderContextMenu(self.indexAt(pos).column(), self)
        menu.popup(self.horizontalHeader().viewport().mapToGlobal(pos))
        
    def getArray(self):
        array = []
        for i in range(self.rowCount()):
            array.append([])
            for j in range(self.columnCount()):
                array[i].append(np.float64(self.item(i, j).text()))
        print(np.array(array))
        return np.array(array)
    
    
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