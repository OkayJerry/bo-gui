from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from collections import OrderedDict


class DoubleDelegate(QItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def createEditor(self, parent, option, index):
        line_edit = QLineEdit(parent)
        line_edit.setValidator(QDoubleValidator())
        return line_edit


class Table(QTableWidget):
    def __init__(self, rows, columns, parent=None):
        super().__init__(rows, columns, parent=parent)
        self.setAlternatingRowColors(True)
        self.setItemDelegate(DoubleDelegate(self))

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.customTableMenuRequested)

        horizontal_header = self.horizontalHeader()
        horizontal_header.setSectionResizeMode(QHeaderView.Stretch)
        horizontal_header.setContextMenuPolicy(Qt.CustomContextMenu)
        horizontal_header.customContextMenuRequested.connect(
            self.customHHeaderMenuRequested)

        vertical_header = self.verticalHeader()
        vertical_header.setContextMenuPolicy(Qt.CustomContextMenu)
        vertical_header.customContextMenuRequested.connect(
            self.customVHeaderMenuRequested)

        for row in range(rows):
            for column in range(columns):
                self.setItem(row, column, DoubleTableItem(''))

    def customHHeaderMenuRequested(self, pos):
        from classes.utility import HorizontalHeaderContextMenu

        menu = HorizontalHeaderContextMenu(self.indexAt(pos).column(), self)
        menu.popup(self.horizontalHeader().viewport().mapToGlobal(pos))

    def customTableMenuRequested(self, pos):
        from classes.utility import TableContextMenu

        menu = TableContextMenu(self.indexAt(pos), self)
        menu.popup(self.viewport().mapToGlobal(pos))

    def customVHeaderMenuRequested(self, pos):
        from classes.utility import VerticalHeaderContextMenu

        menu = VerticalHeaderContextMenu(self.indexAt(pos).row(), self)
        menu.popup(self.viewport().mapToGlobal(pos))

    def getArray(self, empty_last_row=True):
        # creating array version of table
        array = []
        for i in range(self.rowCount()):
            array.append([])
            for j in range(self.columnCount()):
                item = self.item(i, j)
                if not item or item.text() == '':
                    continue
                array[i].append(np.float32(item.text()))

        # removing last row
        if empty_last_row:
            array.pop(-1)

        # return False if table is incomplete
        for row in array:
            if not row or len(row) != self.columnCount():
                return np.array(array), False
        if not array:
            return np.array(array), False

        # else return the successful array
        return np.array(array), True

    def fillFromArray(self, array):
        self.setRowCount(len(array))
        self.setColumnCount(len(array[0]))

        for i, row in enumerate(array):
            for j, val in enumerate(row):
                self.setItem(i, j, DoubleTableItem(val))

    def setHeaderFromTable(self, table):
        for i in range(table.columnCount()):
            header = table.horizontalHeaderItem(i).text()
            self.setHorizontalHeaderItem(i, TableItem(header))

    def setFromTable(self, table):
        array, success = table.getArray()
        if success:
            self.fillFromArray(array)
            self.setHeaderFromTable(table)
        else:
            print('.setFromTable() failed')

    def syncScroll(self, table):
        silder_value = self.verticalScrollBar().value()
        table.verticalScrollBar().setValue(silder_value)


class TableItem(QTableWidgetItem):
    def __init__(self, text: str):
        super().__init__()

        self.setText(text)
        self.setTextAlignment(Qt.AlignCenter)


class DoubleTableItem(TableItem):
    def __init__(self, value: float | str):

        super().__init__(str(value))


class DecisionParameterTable(Table):
    def __init__(self, rows: int, columns: int, parent=None):
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
                self.setItem(self.rowCount() - 1, i, DoubleTableItem(''))

    def reset(self):
        self.setRowCount(0)
        self.setRowCount(1)
        self.setColumnCount(1)
        self.setHorizontalHeaderLabels([f'x0[0]'])


class ObjectiveTable(Table):
    def __init__(self, rows: int, parent=None):
        super().__init__(rows, 1, parent)
        self.setMinimumWidth(150)
        self.setMaximumWidth(150)
        self.setHorizontalHeaderLabels(['y0'])
        self.itemChanged.connect(self.onChange)

    def reset(self):
        self.setRowCount(0)
        self.setRowCount(1)
        self.setHorizontalHeaderLabels(['y0'])

    def onChange(self, item):
        self.dynamicRow(item)
        self.checkInput(item)

    def dynamicRow(self, item):
        if self.indexFromItem(item).row() == self.rowCount() - 1:
            for i in range(self.columnCount()):
                if not self.item(self.rowCount() - 1, i).text():
                    return

            self.insertRow(self.rowCount())
            for i in range(self.columnCount()):
                self.setItem(self.rowCount() - 1, i, DoubleTableItem(''))

            self.updateNumEvaluationData()

    def updateNumEvaluationData(self):
        import globals as glb

        num = (self.rowCount() - 1) * self.columnCount()
        glb.main_window.num_obj_data_label.setText(str(num))

    def checkInput(self, item):
        if item.text() == '':
            return

        bounds = [-20, 20]
        val = float(item.text())
        if val < bounds[0] or val > bounds[1]:  # if val not in bounds
            QMessageBox.warning(self,
                                'WARNING',
                                f'It is recommended to have the objective value bounded between {bounds[0]} and {bounds[1]} for numerical stability.',
                                QMessageBox.Ok,
                                QMessageBox.Ok)


class BoundryTable(Table):
    def __init__(self, columns: int, parent=None):
        super().__init__(2, columns, parent)
        self.setMaximumHeight(85)
        self.setMinimumHeight(85)
        self.setHorizontalHeaderLabels([f'x0[{i}]' for i in range(columns)])
        self.setVerticalHeaderLabels(['min', 'max'])
        self.verticalHeader().setContextMenuPolicy(Qt.NoContextMenu)

    def reset(self):
        self.setColumnCount(1)
        self.setHorizontalHeaderLabels([f'x0[0]'])
        for row in range(self.rowCount()):
            self.item(row, 0).setText('')

    def getBounds(self):
        bounds = []
        for i in range(self.columnCount()):
            pair = []
            for j in range(self.rowCount()):
                item = self.item(j, i)
                if not item or item.text() == '':
                    return bounds, False
                val = np.float64(item.text())
                pair.append(val)
            bounds.append(tuple(pair))
        return bounds, True


class EvaluationPointTable(Table):
    def __init__(self, columns: int, parent=None):
        super().__init__(0, columns, parent=parent)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setContextMenuPolicy(Qt.NoContextMenu)
        self.verticalHeader().setContextMenuPolicy(Qt.NoContextMenu)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)

    def addEvaluationX(self, evaluation_x):
        self.setRowCount(0)
        self.setColumnCount(len(evaluation_x[0]))

        for i, row in enumerate(evaluation_x):
            self.insertRow(i)
            for j, val in enumerate(row):
                self.setItem(i, j, DoubleTableItem(val))

    def setFromTable(self, table):
        self.setColumnCount(table.columnCount())
        self.setHeaderFromTable(table)


class AcqPostFixTable(Table):
    def __init__(self, rows: int, parent=None):
        super().__init__(rows, 1, parent=parent)
        self.setHorizontalHeaderItem(0, TableItem('Fix Value'))

    def returnAxis(self, axis_name: str):
        for row_n in range(self.rowCount()):
            if axis_name == self.verticalHeaderItem(row_n).text():
                self.showRow(row_n)
                break

    def getAxisIndex(self, axis_name: str):
        for row_n in range(self.rowCount()):
            if axis_name == self.verticalHeaderItem(row_n).text():
                return row_n

    def getFixedDimValues(self):
        d = {}
        for row_n in range(self.rowCount()):
            if not self.isRowHidden(row_n):
                try:
                    d[row_n] = np.float32(self.item(row_n, 0).text())
                except:
                    d[row_n] = np.float32(0.0)

        return d

class RampingWeightTable(Table):
    def __init__(self, rows, parent=None):
        super().__init__(rows, 1, parent=parent)
        self.setHorizontalHeaderItem(0, TableItem('Ramping Rate'))
        self.setVerticalHeaderLabels([f'x0[{i}]' for i in range(rows)])
        self.horizontalHeader().setContextMenuPolicy(Qt.NoContextMenu)
        self.verticalHeader().setContextMenuPolicy(Qt.NoContextMenu)

        for i in range(rows):
            self.setItem(i, 0, DoubleTableItem(1))
        
    def getWeights(self):
        weights = []
        for row_n in range(self.rowCount()):
            val = np.float32(self.item(row_n, 0).text())
            weights.append(val)
        return weights