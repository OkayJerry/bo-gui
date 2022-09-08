from PyQt5.QtWidgets import *

class Table(QTableWidget):
    def __init__(self, rows, columns, parent=None):
        super().__init__(rows, columns, parent=parent)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
