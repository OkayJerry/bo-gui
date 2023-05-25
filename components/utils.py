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


class BaseCellContextMenu(QMenu):
    def cut(self):
        """Cuts selected items as their text from the parent QTableWidget."""
        s = ""
        for selection_range in self.parent().selectedRanges():
            for row in range(selection_range.topRow(), selection_range.bottomRow() + 1):
                for column in range(selection_range.leftColumn(), selection_range.rightColumn() + 1):
                    item = self.parent().item(row, column)
                    s += item.text() if item else ''
                    s += '\t' if column != selection_range.rightColumn() else ''
                    item.setText("") if item else None
                s += '\n'
                
        cb = QApplication.clipboard()
        cb.setText(s, mode=cb.Clipboard)
    def copy(self):
        """Copies selected items as their text from the parent QTableWidget."""
        s = ""
        for selection_range in self.parent().selectedRanges():
            for row in range(selection_range.topRow(), selection_range.bottomRow() + 1):
                for column in range(selection_range.leftColumn(), selection_range.rightColumn() + 1):
                    item = self.parent().item(row, column)
                    s += item.text() if item else ''
                    s += '\t' if column != selection_range.rightColumn() else ''
                s += '\n'

        cb = QApplication.clipboard()
        cb.setText(s, mode=cb.Clipboard)
    def paste(self):
        """
        Pastes items from clipboard into table.
        
        Uses the same format as Microsoft Excel, making them cross-compatible. Rows
        are separated by a newline character and columns are separated by a tab
        character.
        """
        cb = QApplication.clipboard()
        s = cb.text()
        
        rows = s.split('\n')
        start_row, start_column = self.parent().currentRow(), self.parent().currentColumn()
        for row_index, row in enumerate(rows):
            if row.strip() and row_index + start_row < self.parent().rowCount():
                columns = row.split('\t')
                for column_index, value in enumerate(columns):
                    if column_index + start_column < self.parent().columnCount():
                        item = CenteredTableItem(value)
                        self.parent().setItem(row_index + start_row, column_index + start_column, item)
    def clear_contents(self):
        """Clears the contents of the currently selected cell(s)."""
        for selection_range in self.parent().selectedRanges():
            for row in range(selection_range.topRow(), selection_range.bottomRow() + 1):
                for column in range(selection_range.leftColumn(), selection_range.rightColumn() + 1):
                    item = self.parent().item(row, column)
                    item.setText("") if item else None
    def insert_row(self, pos: QPoint):
        """Inserts a row to the parent `QTableWidget` at `pos`."""
        model_index = self.parent().indexAt(pos)
        self.parent().insertRow(self.parent().rowCount() if model_index.row() == -1 else model_index.row())
        self.parent().rowInserted.emit(model_index.row())
    def insert_column(self, pos: QPoint):
        """Inserts a column to the parent `QTableWidget` at `pos`."""
        model_index = self.parent().indexAt(pos)
        self.parent().insertColumn(model_index.column())
        self.parent().columnInserted.emit(model_index.column())
    def remove_row(self, pos: QPoint):
        """Removes a row from the parent `QTableWidget` at `pos`."""
        model_index = self.parent().indexAt(pos)
        self.parent().removeRow(model_index.column())
        self.parent().rowRemoved.emit(model_index.column())
    def remove_column(self, pos: QPoint):
        """Removes a column from the parent `QTableWidget` at `pos`."""
        model_index = self.parent().indexAt(pos)
        self.parent().removeColumn(model_index.column())
        self.parent().horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.parent().columnRemoved.emit(model_index.column())


class BaseTable(QTableWidget):
    bottomRowAdded = pyqtSignal(int)
    rowInserted = pyqtSignal(int)
    rowRemoved = pyqtSignal(int)
    columnInserted = pyqtSignal(int)
    columnRemoved = pyqtSignal(int)
    headerChanged = pyqtSignal(int, str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setAlternatingRowColors(True)
        
        self.setItemDelegate(DoubleDelegate(self))  # Allow only numerical keystrokes
        self.itemSelectionChanged.connect(self.verifyCurrentCellExists)
        self.currentItemChanged.connect(self.verifyCurrentCellExists)
    def verifyCurrentCellExists(self):
        if not self.currentItem() or not type(self.currentItem()) is CenteredTableItem:
            self.setItem(self.currentRow(), self.currentColumn(), CenteredTableItem())
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
    def hide_header_by_labels(self, header_labels: List[str], hide_row=True):
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


class XTable(BaseTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MINIMUM_COLUMN_SIZE = 100
        
        self.enableColumnChanges(True)

        self.horizontalHeader().setMinimumSectionSize(MINIMUM_COLUMN_SIZE)
        self.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(self.onHeaderContextMenuRequest)
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.onCellContextMenuRequest)
    def onHeaderContextMenuRequest(self, pos: QPoint) -> None:
        column = self.horizontalHeader().logicalIndexAt(pos)
        context_menu = self.HeaderContextMenu(column, parent=self)
        context_menu.popup(self.horizontalHeader().viewport().mapToGlobal((pos)))
    def onCellContextMenuRequest(self, pos: QPoint) -> None:
        context_menu = self.CellContextMenu(pos, parent=self)
        if not self.allowsColumnChanges():
            context_menu.insert_column_action.setVisible(False)
            context_menu.remove_column_action.setVisible(False)
        context_menu.popup(self.viewport().mapToGlobal(pos))
    def enableColumnChanges(self, b: bool) -> None:
        """Whether the user can insert/remove columns."""
        if type(b) is bool:
            self._allow_column_changes = b
        else:
            raise ValueError(f"`b` must be type `bool`. Got {type(b)}.")
    def allowsColumnChanges(self) -> bool:
        """Whether `self` can insert/remove columns."""
        return self._allow_column_changes
        
    class HeaderContextMenu(QMenu):
        def __init__(self, column: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.column = column
            
            rename_action = QAction("Rename", self)
            rename_action.triggered.connect(self.rename)
            self.addAction(rename_action)
        def rename(self):
            prev_text = self.parent().horizontalHeaderItem(self.column).text()
            new_text, ok = QInputDialog.getText(self.parent(), f"Rename", f"Header:", text=prev_text)
            if ok:
                item = QTableWidgetItem(new_text)
                self.parent().setHorizontalHeaderItem(self.column, item)
                self.parent().headerChanged.emit(self.column, new_text)
    class CellContextMenu(BaseCellContextMenu):
        def __init__(self, pos: QPoint, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
            self.cut_action = QAction("Cut", self)
            self.copy_action = QAction("Copy", self)
            self.paste_action = QAction("Paste", self)
            self.insert_row_action = QAction("Insert Row", self)
            self.insert_column_action = QAction("Insert Column", self)
            self.remove_row_action = QAction("Remove Row", self)
            self.remove_column_action = QAction("Remove Column", self)
            self.clear_contents_action = QAction("Clear Contents", self)
            
            self.cut_action.triggered.connect(self.cut)
            self.copy_action.triggered.connect(self.copy)
            self.paste_action.triggered.connect(self.paste)
            self.insert_row_action.triggered.connect(lambda: self.insert_row(pos))
            self.insert_column_action.triggered.connect(lambda: self.insert_column(pos))
            self.remove_row_action.triggered.connect(lambda: self.remove_row(pos))
            self.remove_column_action.triggered.connect(lambda: self.remove_column(pos))
            self.clear_contents_action.triggered.connect(lambda: self.clear_contents)
            
            self.addAction(self.cut_action)
            self.addAction(self.copy_action)
            self.addAction(self.paste_action)
            self.addSeparator()
            self.addAction(self.insert_row_action)
            self.addAction(self.insert_column_action)
            self.addAction(self.remove_row_action)
            self.addAction(self.remove_column_action)
            self.addAction(self.clear_contents_action)
    
        
class YTable(BaseTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MINIMUM_COLUMN_SIZE = 100

        self.horizontalHeader().setMinimumSectionSize(MINIMUM_COLUMN_SIZE)
        self.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(self.onHeaderContextMenuRequest)
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect()
        
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
    def onHeaderContextMenuRequest(self, pos: QPoint) -> None:
        column = self.horizontalHeader().logicalIndexAt(pos)
        context_menu = self.HeaderContextMenu(column, parent=self)
        context_menu.popup(self.horizontalHeader().viewport().mapToGlobal((pos)))
    def onCellContextMenuRequest(self, pos: QPoint) -> None:
        context_menu = self.CellContextMenu(pos, parent=self)
        context_menu.popup(self.viewport().mapToGlobal(pos))
        
    class HeaderContextMenu(QMenu):
        def __init__(self, column: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.column = column
            
            rename_action = QAction("Rename", self)
            rename_action.triggered.connect(self.rename)
            self.addAction(rename_action)
        def rename(self):
            prev_text = self.parent().horizontalHeaderItem(self.column).text()
            new_text, ok = QInputDialog.getText(self.parent(), f"Rename", f"Header:", text=prev_text)
            if ok:
                item = QTableWidgetItem(new_text)
                self.parent().setHorizontalHeaderItem(self.column, item)
                self.parent().headerChanged.emit(self.column, new_text)
    class CellContextMenu(BaseCellContextMenu):
        def __init__(self, pos: QPoint, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
            self.cut_action = QAction("Cut", self)
            self.copy_action = QAction("Copy", self)
            self.paste_action = QAction("Paste", self)
            self.insert_row_action = QAction("Insert Row", self)
            self.remove_row_action = QAction("Remove Row", self)
            self.clear_contents_action = QAction("Clear Contents", self)
            
            self.cut_action.triggered.connect(self.cut)
            self.copy_action.triggered.connect(self.copy)
            self.paste_action.triggered.connect(self.paste)
            self.insert_row_action.triggered.connect(lambda: self.insert_row(pos))
            self.remove_row_action.triggered.connect(lambda: self.remove_row(pos))
            self.clear_contents_action.triggered.connect(lambda: self.clear_contents)
            
            self.addAction(self.cut_action)
            self.addAction(self.copy_action)
            self.addAction(self.paste_action)
            self.addSeparator()
            self.addAction(self.insert_row_action)
            self.addAction(self.remove_row_action)
            self.addAction(self.clear_contents_action)

    
class BoundryTable(BaseTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MINIMUM_COLUMN_SIZE = 100
        
        self.horizontalHeader().setMinimumSectionSize(MINIMUM_COLUMN_SIZE)
        self.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect()
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect()
    
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
    def onHeaderContextMenuRequest(self, pos: QPoint) -> None:
        column = self.horizontalHeader().logicalIndexAt(pos)
        context_menu = self.HeaderContextMenu(column, parent=self)
        context_menu.popup(self.horizontalHeader().viewport().mapToGlobal((pos)))
    def onCellContextMenuRequest(self, pos: QPoint) -> None:
        context_menu = self.CellContextMenu(pos, parent=self)
        context_menu.popup(self.viewport().mapToGlobal(pos))
    
    class HeaderContextMenu(QMenu):
        def __init__(self, column: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.column = column
            
            rename_action = QAction("Rename", self)
            rename_action.triggered.connect(self.rename)
            self.addAction(rename_action)
        def rename(self):
            prev_text = self.parent().horizontalHeaderItem(self.column).text()
            new_text, ok = QInputDialog.getText(self.parent(), f"Rename", f"Header:", text=prev_text)
            if ok:
                item = QTableWidgetItem(new_text)
                self.parent().setHorizontalHeaderItem(self.column, item)
                self.parent().headerChanged.emit(self.column, new_text) 
    class CellContextMenu(BaseCellContextMenu):
        def __init__(self, pos: QPoint, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
            self.cut_action = QAction("Cut", self)
            self.copy_action = QAction("Copy", self)
            self.paste_action = QAction("Paste", self)
            self.insert_column_action = QAction("Insert Column", self)
            self.remove_column_action = QAction("Remove Column", self)
            self.clear_contents_action = QAction("Clear Contents", self)
            
            self.cut_action.triggered.connect(self.cut)
            self.copy_action.triggered.connect(self.copy)
            self.paste_action.triggered.connect(self.paste)
            self.insert_column_action.triggered.connect(lambda: self.insert_column(pos))
            self.remove_column_action.triggered.connect(lambda: self.remove_column(pos))
            self.clear_contents_action.triggered.connect(lambda: self.clear_contents)
            
            self.addAction(self.cut_action)
            self.addAction(self.copy_action)
            self.addAction(self.paste_action)
            self.addSeparator()
            self.addAction(self.insert_column_action)
            self.addAction(self.remove_column_action)
            self.addAction(self.clear_contents_action)


class CandidateTable(BaseTable):
    selectedToDecisionRequested = pyqtSignal()
    allToDecisionRequested = pyqtSignal()
    selectedToPendingRequested = pyqtSignal()
    allToPendingRequested = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MINIMUM_COLUMN_SIZE = 50
        
        self.horizontalHeader().setMinimumSectionSize(MINIMUM_COLUMN_SIZE)
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect()
        
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
    def onCellContextMenuRequest(self, pos: QPoint) -> None:
        context_menu = self.CellContextMenu(parent=self)
        context_menu.popup(self.viewport().mapToGlobal(pos))
    
    class CellContextMenu(BaseCellContextMenu):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
            self.move_all_menu = self.addMenu("Move All")
            self.move_selected_menu = self.addMenu("Move Selected")
            
            self.all_to_decision = QAction("To Decision", self)
            self.all_to_pending = QAction("To Pending")
            self.selected_to_decision = QAction("To Decision")
            self.selected_to_pending = QAction("To Pending")
            self.copy_action = QAction("Copy", self)
            
            self.all_to_decision.triggered.connect(self.allToDecisionRequested.emit)
            self.all_to_pending.triggered.connect(self.allToPendingRequested.emit)
            self.selected_to_decision.triggered.connect(self.selectedToDecisionRequested.emit)
            self.selected_to_pending.triggered.connect(self.selectedToPendingRequested.emit)
            self.copy_action.triggered.connect(self.copy)

            self.move_all_menu.addAction(self.all_to_decision)
            self.move_all_menu.addAction(self.all_to_pending)
            self.move_selected_menu.addAction(self.selected_to_decision)
            self.move_selected_menu.addAction(self.selected_to_pending)
            self.addSeparator()
            self.addAction(self.copy_action)


class PendingTable(BaseTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MINIMUM_COLUMN_SIZE = 50
        
        self.horizontalHeader().setMinimumSectionSize(MINIMUM_COLUMN_SIZE)
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.onCellContextMenuRequest)
    def onCellContextMenuRequest(self, pos: QPoint) -> None:
        context_menu = self.CellContextMenu(pos, parent=self)
        context_menu.popup(self.viewport().mapToGlobal(pos))
        
    class CellContextMenu(BaseCellContextMenu):
        def __init__(self, pos: QPoint, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
            self.cut_action = QAction("Cut", self)
            self.copy_action = QAction("Copy", self)
            self.paste_action = QAction("Paste", self)
            self.insert_row_action = QAction("Insert Row", self)
            self.remove_row_action = QAction("Remove Row", self)
            self.clear_contents_action = QAction("Clear Contents", self)
            
            self.cut_action.triggered.connect(self.cut)
            self.copy_action.triggered.connect(self.copy)
            self.paste_action.triggered.connect(self.paste)
            self.insert_row_action.triggered.connect(lambda: self.insert_row(pos))
            self.remove_row_action.triggered.connect(lambda: self.remove_row(pos))
            self.clear_contents_action.triggered.connect(lambda: self.clear_contents)
            
            self.addAction(self.cut_action)
            self.addAction(self.copy_action)
            self.addAction(self.paste_action)
            self.addSeparator()
            self.addAction(self.insert_row_action)
            self.addAction(self.remove_row_action)
            self.addAction(self.clear_contents_action)


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