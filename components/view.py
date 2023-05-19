import os
import sys
from collections import OrderedDict

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from components.plot import *
from components.utils import *
from typing import Callable


class CentralView(QWidget):
    gp_updated = pyqtSignal()
    candidates_queried = pyqtSignal()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.tabs = Tabs()
        self.tabs.addTab(self.parent().main, 'Main')
        self.tabs.addTab(self.parent().plots, 'Plots')
        layout.addWidget(self.tabs)
        
class MainView(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Group Boxes
        display_groupbox = QGroupBox('Display')
        display_layout = QGridLayout()
        display_groupbox.setLayout(display_layout)
        
        plots_groupbox = QGroupBox('Preview Plots')
        plots_layout = QVBoxLayout()
        plots_groupbox.setLayout(plots_layout)
        
        control_groupbox = QGroupBox('Control')
        control_layout = QGridLayout()
        control_groupbox.setLayout(control_layout)
        
        data_groupbox = QGroupBox('Data')
        data_layout = QGridLayout()
        data_groupbox.setLayout(data_layout)
        
        query_groupbox = QGroupBox('Query')
        query_layout = QGridLayout()
        query_groupbox.setLayout(query_layout)
        
        # Display
        self.epoch_label = ModularLabel('0')
        
        display_layout.addWidget(QLabel('Iteration Epoch:'), 
                                 0, 0, 1, 1, alignment=Qt.AlignRight)
        display_layout.addWidget(self.epoch_label, 
                                 0, 1, 1, 1)

        # Preview Plots
        self.plot_button = QCheckBox()
        self.canvas = PreviewCanvas()
        
        plot_button_qwidget = QWidget()
        plot_button_layout = QHBoxLayout()
        plot_button_qwidget.setLayout(plot_button_layout)
        plot_button_layout.setContentsMargins(0, 0, 0, 0)
        plot_button_layout.addWidget(QLabel('Draw Plot:'),
                                     alignment=Qt.AlignRight)
        plot_button_layout.addWidget(self.plot_button,
                                     alignment=Qt.AlignLeft)
        
        plots_layout.addWidget(plot_button_qwidget)
        plots_layout.addWidget(self.canvas)
        plots_layout.setStretchFactor(plot_button_qwidget, 0)
        plots_layout.setStretchFactor(self.canvas, 2)
        
        # Control
        self.ucb_button = QRadioButton('Upper Confidence Bound (UCB)')
        self.kg_button = QRadioButton('Knowledge Gradient (KG)')
        self.batch_spin_box = QSpinBox()
        self.beta_spinbox = QDoubleSpinBox()
        
        line = QFrame()
        line.setGeometry(QRect(320, 150, 118, 3))
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        control_layout.addWidget(QLabel('Batch Size:'), 
                                 0, 0, 1, 1, alignment=Qt.AlignRight)
        control_layout.addWidget(self.batch_spin_box, 
                                 0, 1, 1, 2)
        control_layout.addWidget(line,
                                 1, 0, 1, 4)
        control_layout.addWidget(self.ucb_button, 
                             2, 0, 1, 3)
        control_layout.addWidget(QLabel('Beta:'), 
                             3, 0, 1, 1, alignment=Qt.AlignRight)
        control_layout.addWidget(self.beta_spinbox, 
                             3, 1, 1, 3)
        control_layout.addWidget(self.kg_button, 
                             4, 0, 1, 3)
        
        
        
        # Data
        self.x_table = XTable()
        self.y_table = YTable()
        self.boundry_table = BoundryTable(1, 1)  # arbitrary dimensions, but necessary for row height
        self.obj_func_btn = QPushButton('Apply Objective Function')
        self.obj_func_win = CustomFunctionWindow()

        data_layout.addWidget(QLabel('Decision Parameters'), 
                              0, 0, 1, 3, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.x_table,
                              1, 0, 2, 3)
        data_layout.addWidget(QLabel('Objective'),
                              0, 3, 1, 1, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.y_table, 
                              1, 3, 1, 1)
        data_layout.addWidget(self.obj_func_btn, 
                              2, 3, 1, 1)
        data_layout.addWidget(QLabel('Boundary'),
                              3, 0, 1, 3, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.boundry_table, 
                              4, 0, 1, 3)
        data_layout.setColumnMinimumWidth(0, 600)
        data_layout.setColumnStretch(0, 1)

        
        
        # Query
        self.candidate_pnt_table = CandidateTable()
        self.pending_pnt_table = PendingTable()
        self.query_candidates_button = QPushButton('Query Candidates')
        
        query_layout.addWidget(QLabel('Candidate Points'), 
                               0, 0, 1, 1, alignment=Qt.AlignCenter)
        query_layout.addWidget(self.candidate_pnt_table, 
                               1, 0, 1, 1)
        query_layout.addWidget(QLabel('Pending Points'), 
                               0, 1, 1, 1, alignment=Qt.AlignCenter)
        query_layout.addWidget(self.pending_pnt_table, 
                               1, 1, 1, 1)
        query_layout.addWidget(self.query_candidates_button,
                               2, 0, 1, 1)

        # Organization
        top_right_qwidget = QWidget()
        top_right_layout = QHBoxLayout()
        top_right_layout.setContentsMargins(0, 0, 0, 0)
        top_right_qwidget.setLayout(top_right_layout)
        top_right_layout.addWidget(control_groupbox, 1)
        top_right_layout.addWidget(query_groupbox, 3)

        left_qwidget = QWidget()
        right_qwidget = QWidget()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setContentsMargins(0, 0, 0, 0)
        left_qwidget.setLayout(left_layout)
        right_qwidget.setLayout(right_layout)
        
        self.update_gp_button = QPushButton('Update GP')
        
        self.progress_button = ProgressButton(self.update_gp_button)
        
        left_layout.addWidget(display_groupbox, 1)
        left_layout.addWidget(plots_groupbox, 5)
        right_layout.addWidget(top_right_qwidget, 1)
        right_layout.addWidget(data_groupbox, 5)
        right_layout.addWidget(self.progress_button, 1)
        left_layout.setStretchFactor(display_groupbox, 0)

        main_layout.addWidget(left_qwidget, 1)
        main_layout.addWidget(right_qwidget, 3)
        
        self.widgets = [self.epoch_label,
                        self.plot_button,
                        self.beta_spinbox,
                        self.kg_button,
                        self.x_table,
                        self.candidate_pnt_table,
                        self.batch_spin_box,
                        self.ucb_button,
                        self.update_gp_button,
                        self.query_candidates_button,
                        self.obj_func_btn,
                        self.obj_func_win,
                        self.y_table,
                        self.pending_pnt_table,
                        self.boundry_table]
        
    def blockWidgetSignals(self, b: bool):
        """
        Disable/enable PyQt widget signals.

        Args:
            b (bool): `True` = disable, `False` = enable
        """
        for widget in self.widgets:
            try:
                widget.blockSignals(b)
            except:
                print(widget)

class PlotsView(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Main Area
        self.canvas = MainCanvas()

        # Group Boxes
        acq_groupbox = QGroupBox('Acquisition')
        acq_layout = QVBoxLayout()
        acq_groupbox.setLayout(acq_layout)
        
        post_groupbox = QGroupBox('Posterior Mean')
        post_layout = QVBoxLayout()
        post_groupbox.setLayout(post_layout)
        
        # Acquisition
        self.acq_min_button = QRadioButton('Project Minimum')  # default
        self.acq_mean_button = QRadioButton('Project Mean')
        self.acq_fix_button = QRadioButton('Project By Fixing')
        self.acq_x_combobox = MemoryComboBox()
        self.acq_y_combobox = MemoryComboBox()
        self.acq_fixed_table = Table()
            
        acq_xy_qwidget = QWidget()        
        acq_xy_layout = QGridLayout()
        acq_xy_qwidget.setLayout(acq_xy_layout)        
        acq_xy_layout.addWidget(QLabel('x-axis:'), 
                                0, 0, 1, 1, alignment=Qt.AlignRight)
        acq_xy_layout.addWidget(self.acq_x_combobox, 
                                0, 1, 1, 6)
        acq_xy_layout.addWidget(QLabel('y-axis:'), 
                                1, 0, 1, 1, alignment=Qt.AlignRight)
        acq_xy_layout.addWidget(self.acq_y_combobox, 
                                1, 1, 1, 6)
        
        acq_layout.addWidget(self.acq_min_button)
        acq_layout.addWidget(self.acq_mean_button)
        acq_layout.addWidget(self.acq_fix_button)
        acq_layout.addWidget(acq_xy_qwidget)
        acq_layout.addWidget(self.acq_fixed_table)

        # Posterior
        self.post_min_button = QRadioButton('Project Minimum')  # default
        self.post_mean_button = QRadioButton('Project Mean')
        self.post_fix_button = QRadioButton('Project By Fixing')
        self.post_x_combobox = MemoryComboBox()
        self.post_y_combobox = MemoryComboBox()
        self.post_fixed_table = Table()
        
        post_xy_qwidget = QWidget()
        post_xy_layout = QGridLayout()
        post_xy_qwidget.setLayout(post_xy_layout)
        post_xy_layout.addWidget(QLabel('x-axis:'), 
                                 0, 0, 1, 1, alignment=Qt.AlignRight)
        post_xy_layout.addWidget(self.post_x_combobox, 
                                 0, 1, 1, 6)
        post_xy_layout.addWidget(QLabel('y-axis:'), 
                                 1, 0, 1, 1, alignment=Qt.AlignRight)
        post_xy_layout.addWidget(self.post_y_combobox, 
                                 1, 1, 1, 6)
        
        post_layout.addWidget(self.post_min_button)
        post_layout.addWidget(self.post_mean_button)
        post_layout.addWidget(self.post_fix_button)
        post_layout.addWidget(post_xy_qwidget)
        post_layout.addWidget(self.post_fixed_table)

        # Create left and right QWidgets for proper resizing
        left_qwidget = QWidget()
        right_qwidget = QWidget()
        
        # Create left and right layouts and add them to their QWidget
        left_layout = QGridLayout()
        right_layout = QVBoxLayout()
        left_qwidget.setLayout(left_layout)
        right_qwidget.setLayout(right_layout)
        
        # Create Navigation Toolbar for plot canvas
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, left_qwidget)

        # Create QSpinBox for plot history (epoch) selection
        self.epoch_spinbox = QSpinBox()
        self.epoch_spinbox.setMinimumWidth(50)
        
        # Create QWidget to group epoch widgets together
        epoch_qwidget = QWidget()
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("Epoch: "), alignment=Qt.AlignRight)
        epoch_layout.addWidget(self.epoch_spinbox)
        epoch_qwidget.setLayout(epoch_layout)
        
        # Create QComboBox for the i-th candidate query
        self.query_combobox = QComboBox()
        
        # Create QWidget to group them together
        query_qwidget = QWidget()
        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("Query: "), alignment=Qt.AlignRight)
        query_layout.addWidget(self.query_combobox)
        query_qwidget.setLayout(query_layout)
        
        # Add the relevant widgets to the left QWidget's layout
        left_layout.addWidget(self.nav_toolbar,
                              0, 0, 1, 2, alignment=Qt.AlignLeft)
        left_layout.addWidget(QWidget(),
                              0, 2, 1, 1)
        left_layout.addWidget(epoch_qwidget,
                              0, 3, 1, 1, alignment=Qt.AlignRight)
        left_layout.addWidget(query_qwidget,
                              0, 4, 1, 1, alignment=Qt.AlignRight)
        left_layout.addWidget(self.canvas,
                              1, 0, 12, 5)
        left_layout.setColumnStretch(2, 10)
        
        # Create the QPushButton for applying updates of fixed projection values
        self.update_proj_button = QPushButton('Update Projection')

        self.progress_button = ProgressButton(self.update_proj_button)
        
        # Add the relevant widgets to the right QWidget's layout
        right_layout.addWidget(acq_groupbox)
        right_layout.addWidget(post_groupbox)
        right_layout.addWidget(self.progress_button)

        # Add the left and right QWidgets to the main layout with proper sizing
        main_layout.addWidget(left_qwidget, 3)
        main_layout.addWidget(right_qwidget, 1)
        
        self.widgets = [self.acq_fixed_table,
                        self.acq_min_button,
                        self.post_fixed_table,
                        self.acq_fix_button,
                        self.acq_x_combobox,
                        self.post_fix_button,
                        self.acq_mean_button,
                        self.acq_y_combobox,
                        self.post_x_combobox,
                        self.update_proj_button,
                        self.query_combobox,
                        self.post_mean_button,
                        self.post_y_combobox,
                        self.epoch_spinbox,
                        self.post_min_button]
        
    def blockWidgetSignals(self, b: bool):
        """
        Disable/enable PyQt widget signals.

        Args:
            b (bool): `True` = disable, `False` = enable
        """
        for widget in self.widgets:
            try:
                widget.blockSignals(b)
            except:
                print(widget)
        
class MenuBar(QMenuBar):
    newFile = pyqtSignal()
    openFile = pyqtSignal()
    saveFile = pyqtSignal()
    saveAsFile = pyqtSignal()
    preferences = pyqtSignal()
    refreshPlots = pyqtSignal()
    priorRequested = pyqtSignal()
    rowCountChangeRequested = pyqtSignal()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        parent = self.parent()
        self.pref_win = PreferencesWindow(parent=self)
        self.prior_win = CustomFunctionWindow(parent=self)
        
        # file menu
        new_action = QAction('&New', parent)
        open_action = QAction('&Open...', parent)
        save_action = QAction('&Save', parent)
        save_as_action = QAction('&Save As...', parent)
        preferences_action = QAction('&Preferences', parent)
        exit_action = QAction('&Exit', parent)
        
        new_action.setShortcut(QKeySequence.New)
        open_action.setShortcut(QKeySequence.Open)
        save_action.setShortcut(QKeySequence.Save)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        preferences_action.setShortcut(QKeySequence.Preferences)
        exit_action.setShortcut(QKeySequence.Quit)
        
        new_action.triggered.connect(self.newFile.emit)
        open_action.triggered.connect(self.openFile.emit)
        # save_action.triggered.connect(self.saveFile.emit)
        save_as_action.triggered.connect(self.saveAsFile.emit)
        preferences_action.triggered.connect(self.preferences.emit)
        exit_action.triggered.connect(sys.exit)

        file_menu = self.addMenu("File")
        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        # file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(preferences_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # edit menu
        undo_action = QAction("&Undo", parent)
        redo_action = QAction("&Redo", parent)
        prior_action = QAction("&Prior Mean", parent)
        row_count_action = QAction("&Row Count", parent)

        prior_action.triggered.connect(self.priorRequested.emit)
        row_count_action.triggered.connect(self.rowCountChangeRequested.emit)

        edit_menu = self.addMenu("Edit")
        edit_menu.addAction(prior_action)
        # edit_menu.addAction(row_count_action)
        
        # view menu
        refresh_action = QAction("&Refresh Plots", parent)
        refresh_action.setShortcut(QKeySequence.Refresh)
        refresh_action.triggered.connect(self.refreshPlots.emit)

        view_menu = self.addMenu("View")
        view_menu.addAction(refresh_action)

class PreferencesWindow(QDialog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # QMenuBar -> Controller
        self.control = self.parent().parent()

        # Default Settings
        self.default = {"App Font Size": 9,
                        "Log Data Path": os.getcwd() + "\log"}
        
        # Widgets
        self.app_font_sb = QSpinBox()
        self.app_font_sb.setRange(4, 24)
        self.app_font_sb.setValue(self.control.preferences.value("App Font Size"))

        app_layout = QGridLayout()
        app_layout.addWidget(QLabel("Font Size"), 0, 0, 1, 1, alignment=Qt.AlignRight)
        app_layout.addWidget(self.app_font_sb, 0, 1, 1, 3)
        
        self.path_log_le = MemoryLineEdit()
        self.path_log_le.setText(self.control.preferences.value("Log Data Path"))

        path_layout = QGridLayout()
        path_layout.addWidget(QLabel("Log Data Path:"), 0, 0, 1, 1, alignment=Qt.AlignRight)
        path_layout.addWidget(self.path_log_le, 0, 1, 1, 3)
        
        # Group Boxes
        app_gb = QGroupBox("Application")
        app_gb.setLayout(app_layout)

        path_gb = QGroupBox("File Pathways")
        path_gb.setLayout(path_layout)
        
        # Finalizing
        self.setWindowTitle("Preferences")
        self.setWindowIcon(QIcon('images/gear.png'))

        layout = QVBoxLayout()
        layout.addWidget(app_gb)
        layout.addWidget(path_gb)
        self.setLayout(layout)
        
class CustomFunctionWindow(QDialog):
    """
    Window for creating a custom function.
    """
    gotCustomFunction = pyqtSignal()  # cannot emit a function
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setWindowTitle("Python Function Editor")
        self.setWhatsThis(
            "This window is used to create a custom function. Notice that the function signature is already defined. You are only responsible for the script that it runs.\n\nThe variable 'X' represents a singular row from the decision parameters, meaning that if you have three dimensions, it will look like 'np.array([val_1, val_2, val_3])'.\n\nIt is required that the script returns types float or int.\n\nYou may import a module as long as this program lies within your PYTHONPATH.\n\nIf you'd prefer to use an IDE, you can save the relevant *.py file in the 'UserFunctions' folder and import it as shown in the template.")
        
        # Create variable to hold the function
        self.custom_function = None

        # Create text edit field for the code (like IDE)
        self.text_edit = QPlainTextEdit()
        
        # Make tabs equal to 4 spaces 
        self.text_edit.setTabStopDistance(QFontMetricsF(
            self.text_edit.font()).horizontalAdvance(' ') * 4)
        
        # Add placeholder/example text
        self.text_edit.setPlainText(
            "from UserFunctions.examples import rosenbrock\n\nreturn rosenbrock(X)")
        
        # Create button to indicate completion
        self.done_btn = QPushButton("Done")
        
        # Hide window
        self.done_btn.clicked.connect(self._done)
        
        # Initialize layout and fill it with the relevant widgets
        layout = QGridLayout()
        layout.addWidget(QLabel("def function(X):"),
                         0, 0, 1, 1, Qt.AlignLeft)
        layout.addWidget(self.text_edit,
                         1, 0, 1, 2)
        layout.addWidget(self.done_btn,
                         2, 1, 1, 1)
        
        # Set layout
        self.setLayout(layout)
    def _done(self):
        self.hide()
        self.gotCustomFunction.emit()
    

class TableContextMenu(QMenu):
    """
    Base class for a QTableWidget's context menu.
    """
    def __init__(self, include_actions=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.cut_action = QAction('Cut', self)
        self.copy_action = QAction('Copy', self)
        self.paste_action = QAction('Paste', self)
        self.clear_contents_action = QAction('Clear Contents', self)
        
        self.cut_action.triggered.connect(lambda: self.cut(self.parent()))
        self.copy_action.triggered.connect(lambda: self.copy(self.parent()))
        self.paste_action.triggered.connect(lambda: self.paste(self.parent()))
        self.clear_contents_action.triggered.connect(lambda: self.clear_contents(self.parent()))
        
        if include_actions:
            self.addAction(self.cut_action)
            self.addAction(self.copy_action)
            self.addAction(self.paste_action)
            self.addSeparator()
            self.addAction(self.clear_contents_action)
    def cut(self, table: QTableWidget):
        """Cuts selected items as their text from QTableWidget.

        Args:
            table (QTableWidget): Parent table.
        """
        s = ""
        for selection_range in table.selectedRanges():
            for row in range(selection_range.topRow(), selection_range.bottomRow() + 1):
                for column in range(selection_range.leftColumn(), selection_range.rightColumn() + 1):
                    item = table.item(row, column)
                    s += item.text() if item else ''
                    s += '\t' if column != selection_range.rightColumn() else ''
                    item.setText("") if item else None
                s += '\n'
                
        cb = QApplication.clipboard()
        cb.setText(s, mode=cb.Clipboard)
    def copy(self, table: QTableWidget):
        """Copies selected items as their text from QTableWidget.

        Args:
            table (QTableWidget): Parent table.
        """
        s = ""
        for selection_range in table.selectedRanges():
            for row in range(selection_range.topRow(), selection_range.bottomRow() + 1):
                for column in range(selection_range.leftColumn(), selection_range.rightColumn() + 1):
                    item = table.item(row, column)
                    s += item.text() if item else ''
                    s += '\t' if column != selection_range.rightColumn() else ''
                s += '\n'

        cb = QApplication.clipboard()
        cb.setText(s, mode=cb.Clipboard)
    def paste(self, table: QTableWidget):
        """Pastes items from clipboard into table.
        
        Uses the same format as Microsoft Excel, making them cross-compatible. Rows
        are separated by a newline character and columns are separated by a tab
        character.

        Args:
            table (QTableWidget): Parent table.
        """
        cb = QApplication.clipboard()
        s = cb.text()
        
        rows = s.split('\n')
        start_row, start_column = table.currentRow(), table.currentColumn()
        for row_index, row in enumerate(rows):
            if row.strip() and row_index + start_row < table.rowCount():
                columns = row.split('\t')
                for column_index, value in enumerate(columns):
                    if column_index + start_column < table.columnCount():
                        item = CenteredTableItem(value)
                        table.setItem(row_index + start_row, column_index + start_column, item)
    def clear_contents(self, table: QTableWidget):
        for selection_range in table.selectedRanges():
            for row in range(selection_range.topRow(), selection_range.bottomRow() + 1):
                for column in range(selection_range.leftColumn(), selection_range.rightColumn() + 1):
                    item = table.item(row, column)
                    item.setText("") if item else None
    def insert_row(self):
        model_index = self.parent().indexAt(self._pos)
        self.parent().insertRow(self.parent().rowCount() if model_index.row() == -1 else model_index.row())
        self.parent().rowInserted.emit(model_index.row())
    def insert_column(self):
        model_index = self.parent().indexAt(self._pos)
        self.parent().insertColumn(model_index.column())
        self.parent().columnInserted.emit(model_index.column())
    def remove_row(self):
        model_index = self.parent().indexAt(self._pos)
        self.parent().removeRow(model_index.column())
        self.parent().rowRemoved.emit(model_index.column())
    def remove_column(self):
        model_index = self.parent().indexAt(self._pos)
        self.parent().removeColumn(model_index.column())
        self.parent().horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.parent().columnRemoved.emit(model_index.column())
class StandardTableContextMenu(TableContextMenu):
    def __init__(self, pos: QPoint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.removeAction(self.clear_contents_action)

        self._pos = pos
                
        insert_submenu = QMenu("Insert", self)
        insert_row_action = QAction('Row', self)
        self.insert_column_action = QAction("Column", self)
        insert_submenu.addActions([insert_row_action, self.insert_column_action])

        remove_submenu = QMenu("Remove", self)
        remove_row_action = QAction('Row', self)
        self.remove_column_action = QAction("Column", self)
        remove_submenu.addActions([remove_row_action, self.remove_column_action])

        insert_row_action.triggered.connect(self.insert_row)
        self.insert_column_action.triggered.connect(self.insert_column)
        remove_row_action.triggered.connect(self.remove_row)
        self.remove_column_action.triggered.connect(self.remove_column)
        
        if isinstance(self.parent(), YTable):
            self.insert_column_action.setVisible(False)
            self.remove_column_action.setVisible(False)
        
        self.addSeparator()
        self.addMenu(insert_submenu)
        self.addMenu(remove_submenu)
        self.addAction(self.clear_contents_action)
        

class HeaderContextMenu(QMenu):
    def __init__(self, column, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._column = column

        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(self._rename)
        self.addAction(rename_action)

    def _rename(self):
        prev = self.parent().horizontalHeaderItem(self._column).text()
        text, ok = QInputDialog.getText(self.parent(), f"Rename", f"Header:", text=prev)
        if ok:
            item = QTableWidgetItem(text)
            self.parent().setHorizontalHeaderItem(self._column, item)
            self.parent().headerChanged.emit(self._column, text)

class CandidatePendingTableContextMenu(TableContextMenu):
    def __init__(self, pos: QPoint, *args, **kwargs):
        super().__init__(include_actions=False, *args, **kwargs)
        
        self._pos = pos
        parent_table = self.parent()

        selected_to_decision_action = QAction("To Decision", self)
        all_to_decision_action = QAction("To Decision", self)

        all_to_decision_action.triggered.connect(parent_table.allToX)
        selected_to_decision_action.triggered.connect(parent_table.selectedToX)

        move_selected_menu = self.addMenu("Move Selected")
        move_all_menu = self.addMenu("Move All")
        
        move_selected_menu.addAction(selected_to_decision_action)
        move_all_menu.addAction(all_to_decision_action)
        
        self.addSeparator()
        self.addAction(self.cut_action)
        self.addAction(self.copy_action)
        self.addAction(self.paste_action)
        self.addSeparator()
        
        if isinstance(self.parent(), CandidateTable):
            all_to_other_action = QAction("To Pending", self)
            selected_to_other_action = QAction("To Pending", self)
            
            all_to_other_action.triggered.connect(parent_table.allToOther)
            selected_to_other_action.triggered.connect(parent_table.selectedToOther)
            
            move_all_menu.addAction(all_to_other_action)
            move_selected_menu.addAction(selected_to_other_action)
        elif isinstance(self.parent(), PendingTable):
            insert_action = QAction("Insert Row", self)
            remove_action = QAction("Remove Row", self)
            
            insert_action.triggered.connect(self.insert_row)
            remove_action.triggered.connect(self.remove_row)
        
            self.addAction(insert_action)
            self.addAction(remove_action)
            self.addAction(self.clear_contents_action)
