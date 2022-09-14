from classes import interactiveGPBO


def init():
    from sys import argv
    from PyQt5.QtWidgets import QApplication
    from classes.windows import MainWindow
    
    global app
    global main_window
    global interactive_GPBO
    
    app = QApplication(argv)
    main_window = MainWindow()