from PyQt5.QtWidgets import *

class HeaderContextMenu(QMenu):
    def __init__(self, parent=None):
        super().__init__(parent)

        rename_action = QAction('Rename', self)

        rename_action.triggered.connect(self.rename)

        self.addAction(rename_action)
        
    def rename(self):
        print('renaming header...')
        