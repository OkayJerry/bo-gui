from classes import interactiveGPBO


def init():
    from sys import argv
    from PyQt5.QtWidgets import QApplication
    from classes.windows import MainWindow
    
    global app
    global main_window
    global interactive_GPBO
    global acq_widgets
    global tables
    
    acq_widgets = []
    tables = []
    interactive_GPBO = None
    
    app = QApplication(argv)
    main_window = MainWindow()