def init():
    from sys import argv
    from PyQt5.QtWidgets import QApplication
    from classes.windows import MainWindow
    
    global app
    global main_window
    
    app = QApplication(argv)
    main_window = MainWindow()