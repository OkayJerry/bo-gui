import sys
from PyQt5.QtWidgets import QApplication
from components.controller import Controller as MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(app)
    window.show()
    app.exec_()