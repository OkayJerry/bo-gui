from PyQt5 import QtWidgets, QtGui, QtCore

class DoubleDelegate(QtWidgets.QItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def createEditor(self, parent, option, index):
        # Create a QLineEdit as the editor widget
        line_edit = QtWidgets.QLineEdit(parent)
        
        # Give the QLineEdit a transparent background
        line_edit.setStyleSheet("* { background-color: rgba(200, 200, 200, 255); }")

        # Center text
        line_edit.setAlignment(QtCore.Qt.AlignCenter)
        
        # Reject any non-numerical key strokes
        line_edit.setValidator(QtGui.QDoubleValidator())
        return line_edit
    
    def setEditorData(self, editor, index):
        # Set the text of the editor widget to the value of the item
        value = index.data(QtCore.Qt.EditRole)
        editor.setText(value)

    def setModelData(self, editor, model, index):
        # Set the text of the item to the text of the editor widget
        text = editor.text()
        model.setData(index, text, QtCore.Qt.EditRole)