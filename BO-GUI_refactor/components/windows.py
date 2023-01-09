from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from components.pages import *
from components.menus import MenuBar

from components.tables import set_row_count

#############
# Functions #
#############



##############
# Subclasses #
##############

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set the window title, icon, and size
        self.setWindowTitle('BO-GUI')
        self.setWindowIcon(QIcon('images/frib.png'))
        self.resize(640, 480)

        # Create the central widget and set it as the main window's central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create the main layout for the central widget
        central_layout = QVBoxLayout()
        central_widget.setLayout(central_layout)

        # Create the menu bar and set it as the main window's menu bar
        self.menu_bar = MenuBar(parent=self)
        self.setMenuBar(self.menu_bar)

        # Create the status bar and set it as the main window's status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create a QTabWidget for navigation and add it to the central widget
        self.tabs = QTabWidget()
        central_layout.addWidget(self.tabs)
        
        # Create the different pages for the QTabWidget
        self.initialization_page = InitializationPage()
        self.iteration_page = IterationPage()
        self.plots_page = PlotsPage()
        
        self.tabs.addTab(self.initialization_page, 'Initialization')
        self.tabs.addTab(self.iteration_page, 'Iteration')
        self.tabs.addTab(self.plots_page, 'Plots')

        # Connect widgets together (signals)
        self.connectWidgets()

    def connectWidgets(self):
        # Sync addition of bottom row for initial XY table pair
        init_x_table = self.initialization_page.x_table
        init_y_table = self.initialization_page.y_table
        init_x_table.bottomRowAdded.connect(lambda row_count: 
            set_row_count(row_count, init_y_table))
        init_y_table.bottomRowAdded.connect(lambda row_count: 
            set_row_count(row_count, init_x_table))
        
        # Sync addition of bottom row for iteration XY table pair
        iter_x_table = self.iteration_page.x_table
        iter_y_table = self.iteration_page.y_table
        iter_x_table.bottomRowAdded.connect(lambda row_count: 
            set_row_count(row_count, iter_y_table))
        iter_y_table.bottomRowAdded.connect(lambda row_count: 
            set_row_count(row_count, iter_x_table))
        
        # Enable initial beta spin box for 'UCB' radio button only
        ucb_button = self.initialization_page.ucb_button
        kg_button = self.initialization_page.kg_button
        beta_spinbox = self.initialization_page.beta_spinbox
        
        ucb_button.clicked.connect(lambda: beta_spinbox.setEnabled(True))
        kg_button.clicked.connect(lambda: beta_spinbox.setEnabled(False))
        
        
        