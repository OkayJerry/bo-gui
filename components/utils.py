from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from typing import List, Tuple, Union, Dict
from collections import OrderedDict

class EmptyStringDoubleValidator(QDoubleValidator):
    def validate(self, input_str, pos):
        if input_str == '':
            return (QDoubleValidator.Acceptable, '', pos)
        return super().validate(input_str, pos)

class DoubleDelegate(QItemDelegate):
    def createEditor(self, parent, option, index):
        line_edit = QLineEdit(parent)
        line_edit.setStyleSheet("* { background-color: rgba(200, 200, 200, 255); }")
        line_edit.setAlignment(Qt.AlignCenter)
        line_edit.setValidator(EmptyStringDoubleValidator())
        return line_edit
    
    def setEditorData(self, editor, index):
        value = index.data(Qt.EditRole)
        editor.setText(value)

    def setModelData(self, editor, model, index):
        text = editor.text()
        model.setData(index, text, Qt.EditRole)
        
class CenteredTableItem(QTableWidgetItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setTextAlignment(Qt.AlignCenter)

class Table(QTableWidget):    
    bottomRowAdded = pyqtSignal(int)
    rowInserted = pyqtSignal(int)
    rowRemoved = pyqtSignal(int)
    columnInserted = pyqtSignal(int)
    columnRemoved = pyqtSignal(int)
    headerChanged = pyqtSignal(int, str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Allow only numerical keystrokes
        self.setItemDelegate(DoubleDelegate(self))

    def to_list(self) -> List[List[Union[float, None]]]:
        """Converts the table into a 2-dimensional list.

        Returns:
            List: List of lists (array).
        """
        data = []
        for row in range(self.rowCount()):
            data_row = []
            for column in range(self.columnCount()):
                item = self.item(row, column)
                data_row.append(float(item.text()) if item and item.text() != "" else None)
            data.append(data_row)

        return data
    def to_dim(self) -> Dict[int, float]:
        d = {}
        
        for row_n in range(self.rowCount()):
            if not self.isRowHidden(row_n):
                item = self.item(row_n, 0)
                d[row_n] = float(item.text()) if item.text() != "" else 0.0
        return d
    def fill(self, data: List[List[float]]) -> None:
        """
        Fills the table with 2-dimensional data.

        Args:
            data (List): List of lists of data.
        """
        self.setColumnCount(len(data[0]))

        for row, row_data in enumerate(data):
            for column, cell_data in enumerate(row_data):
                item = CenteredTableItem(str(cell_data))
                self.setItem(row, column, item)
    def append_items(self, items: List[QTableWidgetItem]) -> None:
        """
        Appends list of items to table.
    
        IMPORTANT: APPENDS FROM LEFT TO RIGHT, TOP TO BOTTOM ON TABLE.
        This means that items will not maintain their original placement.

        Args:
            items (list): List of items to append.
            table (QTableWidget): Table to append items to.
        """
        row_all_none = lambda row: all(item is None or item.text() is "" for item in [self.item(row, i) for i in range(self.columnCount())])

        for i in range(self.rowCount()):
            if row_all_none(i) and len(items) > 0:
                for j in range(self.columnCount()):
                    print(f"({i},{j})")
                    self.setItem(i, j, CenteredTableItem() if items[0] is None else items.pop(0))
                
        if len(items) > 0:
            rows_needed: int = len(items) // self.columnCount()
            
            self.setRowCount(self.rowCount() + rows_needed)
            
            for i, item in enumerate(reversed(items)):
                row_number = self.rowCount() - i // self.columnCount() - 1
                column_number = self.columnCount() - i % self.columnCount() - 1

                self.setItem(row_number, column_number, item)

                
            
    def is_bottom_row_full(self):
        """
        Checks if bottom row is full.

        Returns:
            Bool: True if full, false if not.
        """
        # Get the number of columns in the table
        column_count = self.columnCount()

        # Get the index of the bottom row
        bottom_row_index = self.rowCount() - 1

        # Iterate over the cells in the bottom row and check if they are filled with text
        for i in range(column_count):
            item = self.item(bottom_row_index, i)
            if item is None or item.text() == "":
                return False

        return True
    def remove_empty_rows(self, remove_no_text=True):
        """Removes all empty rows from the table.

        Args:
            remove_no_text (bool, optional): Remove rows with items (or not) but no text. Defaults to True.
        """
        rows_to_remove = []
        for row in range(self.rowCount()):
            is_empty = True
            for col in range(self.columnCount()):
                item = self.item(row, col)
                
                # Conditional remove for more reusability
                if remove_no_text:
                    if item and item.text(): 
                        is_empty = False
                        break
                else:
                    if item: 
                        is_empty = False
                        break
                    
            # If entire row is empty, add it to the removal list
            if is_empty:
                rows_to_remove.append(row)
        
        # Remove in reverse so that other row indices don't shift
        for row in reversed(rows_to_remove):
            self.removeRow(row)
    def remove_by_item(self, items: List[QTableWidgetItem], delete_item=True):
        """Removes each item from a list of items.

        Args:
            items (list): List of items to remove.
            delete_item (bool, optional): Deleting the item will make it impossible to reuse, but saves memory. Defaults to True.
        """
        for item in items:
            row = item.row()
            column = item.column()
            
            # Conditional for more reusability
            if delete_item:
                # .setItem removes the item from memory
                self.setItem(row, column, None)
            else:
                # .takeItem keeps it the item in memory
                self.takeItem(row, column)
    def take_items(self, selected=False):
        if selected:
            items = self.selectedItems()
            self.remove_by_item(items, delete_item=False)
        else:
            items = []
            for i in range(self.rowCount()):
                for j in range(self.columnCount()):
                    item = self.takeItem(i, j)
                    items.append(item)
            
        return items
    def copy_items(self, selected=False) -> List[QTableWidgetItem]:
        if selected:
            items = []
            for item in self.selectedItems():
                text = item.text() if item else ""
                item = CenteredTableItem()
                item.setText(text)
                items.append(item)
        else:
            items = []
            for i in range(self.rowCount()):
                for j in range(self.columnCount()):
                    text = self.item(i, j).text() if self.item(i, j) else ""
                    item = CenteredTableItem()
                    item.setText(text)
                    items.append(item)
        return items
    def fill_row_with_items(self, row: int, item_class):
        """Fills row with items of the provided class.

        Args:
            row (int): The row number to fill.
            item_class (class): Uninstantiated QTableWidgetItem subclass.
        """
        for column in range(self.columnCount()):
            if not self.item(row, column):
                self.setItem(row, column, item_class())
    def set_row_count_and_fill(self, row_count: int, item_class: QTableWidgetItem):
        """Sets row count of table and fills it with items while blocking signals.

        Args:
            row_count (Int): Number of rows.
            item_class (class): Uninstantiated QTableWidgetItem subclass.
        """
        self.blockSignals(True)
        self.setRowCount(row_count)
        
        for row in range(self.rowCount()):
            self.fill_row_with_items(row, item_class)

        self.blockSignals(False)
    def get_horizontal_labels(self) -> List[Union[str, None]]:
        l = []
        for i in range(self.columnCount()):
            item = self.horizontalHeaderItem(i)
            l.append(item.text() if item else None)
        return l
    def get_vertical_labels(self) -> List[Union[str, None]]:
        l = []
        for i in range(self.rowCount()):
            item = self.verticalHeaderItem(i)
            l.append(item.text() if item else None)
        return l
    def show_all_rows(self):
        """Shows all hidden rows of a table.

        Args:
            table (QTableWidget): Table to reveal.
        """
        for row in range(self.rowCount()):
            if self.isRowHidden(row):
                self.setRowHidden(row, False)
    def hide_header_by_labels(self, header_labels: list[str], hide_row=True):
        """Hides row/column based on provided header labels.

        Args:
            table (QTableWidget): Parent table.
            header_labels (list[str]): List of header labels.
            hide_row (bool, optional): Whether to hide row or column. Defaults to True.
        """
        if hide_row:
            for row in range(self.rowCount()):
                if self.verticalHeaderItem(row).text() in header_labels:
                    self.setRowHidden(row, True)
        else:
            for column in range(self.columnCount()):
                if self.horizontalHeaderItem(column).text() in header_labels:
                    self.setColumnHidden(column, True)

class XTable(Table):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().setMinimumSectionSize(100)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        
class YTable(Table):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().setMinimumSectionSize(100)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)

class BoundryTable(Table):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().setMinimumSectionSize(100)
    
        total_row_height = self.rowHeight(0) * 2
        header_height = self.horizontalHeader().height()
        total_frame_width = 2 * self.frameWidth()
        buffer = 2  # for mac
        height = total_row_height + header_height + total_frame_width + buffer
        
        self.setMaximumHeight(height)
        self.setMinimumHeight(height)
        
        self.setVerticalHeaderLabels(['min', 'max'])
        
    def to_list(self) -> List[Tuple[np.float64]]:
        """
        Converts the items of a table into an array with a shape
        eligible to be the GPBO's bounds.

        Args:
            table (QTableWidget): Table to convert.

        Returns:
            list: Bounds in required shape.
        """
        bounds = []
        for i in range(self.columnCount()):
            pair = []
            for j in range(self.rowCount()):
                item = self.item(j, i)
                if not item or item.text() == '':
                    continue
                val = np.float64(item.text())
                pair.append(val)
            bounds.append(tuple(pair))
        return bounds  
        
class CandidateTable(Table):
    selectedToX = pyqtSignal()
    allToX = pyqtSignal()
    selectedToOther = pyqtSignal()
    allToOther = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.horizontalHeader().setMinimumSectionSize(50)

class PendingTable(Table):
    selectedToX = pyqtSignal()
    allToX = pyqtSignal()
    selectedToOther = pyqtSignal()
    allToOther = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().setMinimumSectionSize(50)
        
class MemoryLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous = None
        
class MemoryComboBox(QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._previous: int = None
        
    def clear(self):
        super().clear()
        
        self._previous: int = None
        
    def setPreviousIndex(self, index: int):
        self._previous = index
        
    def previousIndex(self) -> int:
        return self._previous

class Tabs(QTabWidget):
    tabEnabled = pyqtSignal(int, bool)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def setTabEnabled(self, index: int, a1: bool) -> None:
        self.tabEnabled.emit(index, a1)
        super().setTabEnabled(index, a1)

class ProgressButton(QWidget):
    progressBarEnabled = pyqtSignal(bool)

    def __init__(self, button: QPushButton, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        
        self._BUTTON = 0
        self._PROGRESS_BAR = 1
        
        self._button = button
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximum(1019)
        self._progress_bar.valueChanged.connect(self._checkProgressBar)
        
        layout = QStackedLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(button)
        layout.addWidget(self._progress_bar)

        self.setContentsMargins(0, 0, 0, 0)
        self.setMaximumHeight(25)
        self.setLayout(layout)
        
    def enableProgressBar(self):
        self.layout().setCurrentIndex(self._PROGRESS_BAR) 
        print("$$")
        self.progressBarEnabled.emit(True)
        print("##")
        
    def disableProgressBar(self):
        self.layout().setCurrentIndex(self._BUTTON)
        self.progressBarEnabled.emit(False)
        
    def _checkProgressBar(self, value: int):
        if value >= self._progress_bar.maximum():
            self.reset()

    def reset(self) -> None:
        self.disableProgressBar()
        self._progress_bar.reset()

    def updateValue(self, value: int) -> None:
        self._progress_bar.setValue(self._progress_bar.value() + value)
        
class ModularLabel(QLabel):
    textChanged = pyqtSignal(str)
    def setText(self, a0: str) -> None:
        super().setText(a0)
        self.textChanged.emit(a0)