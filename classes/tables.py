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
                self.setItem(row, column, item)
            
    def customHeaderMenuRequested(self, pos):
        from classes.utility import HeaderContextMenu
        
        menu = HeaderContextMenu(self.indexAt(pos).column(), self)
        menu.popup(self.horizontalHeader().viewport().mapToGlobal(pos))
        
    def getArray(self, empty_row=True):
        if empty_row:
            if self.rowCount() == 1:
                print(self + '$$$$$$$$$$$$$$$$$$$$$$$')
                row_count = 1
            else:
                row_count = self.rowCount() - 1
        else:
            row_count = self.rowCount()

        array = []
        for i in range(row_count):
            array.append([])
            for j in range(self.columnCount()):
                array[i].append(np.float32(self.item(i, j).text()))
        return np.array(array)

    def fillFromArray(self, array):
        self.setRowCount(len(array))
        self.setColumnCount(len(array[0]))

        for i, row in enumerate(array):
            for j, val in enumerate(row):
                item = QTableWidgetItem()
                item.setText(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(i, j, item)
                
    def setHeaderFromTable(self, table):
        for i in range(table.columnCount()):
            header = table.horizontalHeaderItem(i).text()
            
            item = QTableWidgetItem()
            item.setText(header)
            item.setTextAlignment(Qt.AlignCenter)
            self.setHorizontalHeaderItem(i, item)

    def setFromTable(self, table):
        self.fillFromArray(table.getArray())
        self.setHeaderFromTable(table)
            
                
    
    
class DecisionParameterTable(Table):
    def __init__(self, rows, columns, parent=None):
        super().__init__(rows, columns, parent)
        self.setHorizontalHeaderLabels([f'x0[{i}]' for i in range(columns)])
        self.itemChanged.connect(self.dynamicRow)

    def dynamicRow(self, item):
        if self.indexFromItem(item).row() == self.rowCount() - 1:
            for i in range(self.columnCount()):
                item = self.item(self.rowCount() - 1, i)
                if not item or not item.text():
                    return
        
            self.insertRow(self.rowCount())
            for i in range(self.columnCount()):
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(self.rowCount() - 1, i, item)
    
class ObjectiveTable(Table):
    def __init__(self, rows, parent=None):
        super().__init__(rows, 1, parent)
        self.setMinimumWidth(150)
        self.setMaximumWidth(150)
        self.setHorizontalHeaderLabels(['y0'])
        self.itemChanged.connect(self.dynamicRow)

    def dynamicRow(self, item):
        if self.indexFromItem(item).row() == self.rowCount() - 1:
            for i in range(self.columnCount()):
                if not self.item(self.rowCount() - 1, i).text():
                    return
        
            self.insertRow(self.rowCount())
            for i in range(self.columnCount()):
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(self.rowCount() - 1, i, item)
        
class BoundryTable(Table):
    def __init__(self, columns, parent=None):
        super().__init__(2, columns, parent)
        self.setMaximumHeight(85)
        self.setMinimumHeight(85)
        self.setHorizontalHeaderLabels([f'x0[{i}]' for i in range(columns)])
        self.setVerticalHeaderLabels(['min', 'max'])

    def getBounds(self):
        bounds = []
        for i in range(self.columnCount()):
            pair = []
            for j in range(self.rowCount()):
                val = np.float64(self.item(j, i).text())
                pair.append(val)
            bounds.append(tuple(pair))
        return bounds
                
class EvaluationPointTable(Table):
    def __init__(self, columns, parent=None):
        super().__init__(0, columns, parent=parent)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setContextMenuPolicy(Qt.NoContextMenu)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)

    def addEvaluationX(self, evaluation_x):
        self.setRowCount(0)
        for i, row in enumerate(evaluation_x):
            self.insertRow(i)
            for j, val in enumerate(row):
                item = QTableWidgetItem()
                item.setText(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(i, j, item)

    def setFromTable(self, table):
        self.setColumnCount(table.columnCount())
        self.setHeaderFromTable(table)