from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import *

from components.utility import DoubleDelegate

# from PyQt5.QtGui import 


#############
# Functions #
#############

def table_to_array(table):
    # Get the model for the table
    model = table.model()

    # Get the number of rows and columns in the table
    num_rows = model.rowCount()
    num_columns = model.columnCount()

    # Create an empty list to store the data
    data = []

    # Iterate over the rows of the table
    for row in range(num_rows):
        # Create a list to store the data for the current row
        row_data = []

        # Iterate over the columns of the table
        for column in range(num_columns):
            # Get the data for the current cell
            index = model.index(row, column)
            cell_data = model.data(index)

            # Add the cell data to the row data
            row_data.append(cell_data)

        # Add the row data to the overall data
        data.append(row_data)

    # Return the data as an array
    return data

def array_to_table(data, table):
    # Get the model for the table
    model = table.model()

    # Set the number of rows and columns in the table
    model.setRowCount(len(data))
    model.setColumnCount(len(data[0]))

    # Iterate over the rows of the data
    for row, row_data in enumerate(data):
        # Iterate over the columns of the data
        for column, cell_data in enumerate(row_data):
            # Create an item for the cell
            item = QtGui.QStandardItem(cell_data)

            # Set the item for the cell
            model.setItem(row, column, item)
            
def copy_horizontal_header(src_table, dest_table):
    # Get the model for the source table
    src_model = src_table.model()

    # Get the number of columns in the source table
    num_columns = src_model.columnCount()

    # Iterate over the columns of the source table
    for column in range(num_columns):
        # Get the header item for the column
        src_item = src_model.horizontalHeaderItem(column)

        # Set the header item for the same column in the destination table
        dest_table.setHorizontalHeaderItem(column, src_item)

def is_bottom_row_full(table_widget):
    # Get the number of columns in the table
    column_count = table_widget.columnCount()

    # Get the index of the bottom row
    bottom_row_index = table_widget.rowCount() - 1

    # Iterate over the cells in the bottom row and check if they are filled with text
    for i in range(column_count):
        item = table_widget.item(bottom_row_index, i)
        if item is None or item.text() == "":
            return False

    return True


def set_row_count(row_count, table):
    table.blockSignals(True)
    table.setRowCount(row_count)
    table.blockSignals(False)


##############
# Subclasses #
##############

# QTableWidgetItems
# -----------------



# QTableWidget
# ------------

class Table(QTableWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Allow only numerical keystrokes
        self.setItemDelegate(DoubleDelegate(self))
        
class DecisionParameterTable(Table):
    bottomRowAdded = pyqtSignal(int)
    # rowRemoved = pyqtSignal(int)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setHorizontalHeaderLabels([f'x0[{i}]' for i in range(self.columnCount())])
        self.cellChanged.connect(self.onCellChanged)

    def onCellChanged(self, row, column):
        if is_bottom_row_full(self):
            # If all cells in the bottom row are filled with text, add a new row
            self.insertRow(row + 1)

            self.bottomRowAdded.emit(self.rowCount())
            
        
class ObjectiveTable(Table):
    bottomRowAdded = pyqtSignal(int)
    # rowRemoved = pyqtSignal(int)

    def __init__(self, rows: int):
        super().__init__(rows, 1)
        
        self.setHorizontalHeaderLabels(['y0'])
        self.cellChanged.connect(self.onCellChanged)

    def onCellChanged(self, row, column):
        if is_bottom_row_full(self):
            # If all cells in the bottom row are filled with text, add a new row
            self.insertRow(row + 1)

            self.bottomRowAdded.emit(self.rowCount())
            
        
class BoundryTable(Table):
    def __init__(self, columns: int):
        super().__init__(2, columns)
    
        total_row_height = self.rowHeight(0) * self.rowCount()
        header_height = self.horizontalHeader().height()
        total_frame_width = 2 * self.frameWidth()
        buffer = 2  # for mac
        height = total_row_height + header_height + total_frame_width + buffer
        
        self.setMaximumHeight(height)
        self.setMinimumHeight(height)
        
        self.setHorizontalHeaderLabels([f'x0[{i}]' for i in range(self.columnCount())])
        self.setVerticalHeaderLabels(['min', 'max'])
        self.setVerticalHeaderLabels(['min', 'max'])
        
class EvaluationPointTable(Table):
    def __init__(self, columns: int):
        super().__init__(0, columns)

        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        self.setHorizontalHeaderLabels([f'x0[{i}]' for i in range(self.columnCount())])
        self.horizontalHeader().setContextMenuPolicy(Qt.NoContextMenu)
        self.verticalHeader().setContextMenuPolicy(Qt.NoContextMenu)
        
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        
class FixValueTable(Table):
    def __init__(self, rows: int): 
        super().__init__(rows, 1)

        total_row_height = self.rowHeight(0) * self.rowCount()
        header_height = self.horizontalHeader().height()
        total_frame_width = 2 * self.frameWidth()
        buffer = 2  # for mac
        height = total_row_height + header_height + total_frame_width + buffer

        self.setHorizontalHeaderItem(0, QTableWidgetItem('Fix Value'))
        
class RampingWeightTable(Table):
    def __init__(self, rows: int):
        super().__init__(rows, 1)

        self.setHorizontalHeaderItem(0, QTableWidgetItem('Ramping Rate'))
        self.setVerticalHeaderLabels([f'x0[{i}]' for i in range(rows)])
        
        self.horizontalHeader().setContextMenuPolicy(Qt.NoContextMenu)
        self.verticalHeader().setContextMenuPolicy(Qt.NoContextMenu)