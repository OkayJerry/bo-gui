import os

from matplotlib.backends.backend_qtagg import (FigureCanvas,
                                               NavigationToolbar2QT)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

from components.tables import (BoundryTable, DecisionParameterTable,
                               EvaluationPointTable, FixValueTable,
                               ObjectiveTable, RampingWeightTable)


class InitializationPage(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create the main layout and add it to this page
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Create a QGroupBox to organize data-related widgets
        data_groupbox = QGroupBox('Data')
        
        # Create a layout and add it to the QGroupBox
        data_layout = QGridLayout()
        data_groupbox.setLayout(data_layout)
        
        # Create the QTableWidgets for initial parameters
        self.x_table = DecisionParameterTable(1, 3)
        self.y_table = ObjectiveTable(1)
        self.boundry_table = BoundryTable(3)
        
        # Create the QPushButton for automatically calculating objective values
        self.obj_func_btn = QPushButton('Apply Objective Function')

        # Add the relevant widgets to the QGroupBox's layout
        data_layout.addWidget(QLabel('Initial Decision Parameters'), 
                              0, 0, 1, 3, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.x_table,
                              1, 0, 2, 3)
        data_layout.addWidget(QLabel('Initial Objective'),
                              0, 3, 1, 1, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.y_table, 
                              1, 3, 1, 1)
        data_layout.addWidget(self.obj_func_btn, 
                              2, 3, 1, 1)
        data_layout.addWidget(QLabel('Boundary of Decision'),
                              3, 0, 1, 3, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.boundry_table, 
                              4, 0, 1, 3)

        # Formatting layout shape
        data_layout.setColumnMinimumWidth(0, 600)
        data_layout.setColumnStretch(0, 1)

        # Create a QGroupBox to organize acquisition function-related widgets
        acq_groupbox = QGroupBox('Acquisition')
        
        # Create a layout and add it to the QGroupBox 
        acq_layout = QGridLayout()
        acq_groupbox.setLayout(acq_layout)
        
        # Create the QRadioButtons for 'Upper Confidence Bound' and 'Knowledge Gradient'
        self.ucb_button = QRadioButton('Upper Confidence Bound (UCB)')
        self.kg_button = QRadioButton('Knowledge Gradient (KG)')
        
        # Create the QDoubleSpinBox for 'UCB' beta values
        self.beta_spinbox = QDoubleSpinBox()

        # Set initial states
        self.ucb_button.setChecked(True)
        
        # Add the relevant widgets to the QGroupBox's layout
        acq_layout.addWidget(self.ucb_button, 
                             0, 0, 1, 6)
        acq_layout.addWidget(QLabel('Beta:'), 
                             1, 1, 1, 1, alignment=Qt.AlignRight)
        acq_layout.addWidget(self.beta_spinbox, 
                             1, 2, 1, 1)
        acq_layout.addWidget(self.kg_button, 
                             2, 0, 1, 6)

        # Create QGroupBox to organize optional parameter-related widgets
        opt_groupbox = QGroupBox('Optional')
        
        # Create a layout and add it to the QGroupBox
        opt_layout = QGridLayout()
        opt_groupbox.setLayout(opt_layout)
        
        # Create the QTableWidget for ramping rate
        self.weight_table = RampingWeightTable(3)
        
        # Create the QSpinBox for the regularization coefficients
        self.reg_coeff_spinbox = QSpinBox()
        
        # Create the QCheckBox for enabling/disabling bounds corners
        self.bounds_corners_check_box = QCheckBox()
        
        # Create the QLineEdits for various pathways
        self.log_data_path_line_edit = QLineEdit()
        self.prior_mean_model_path_line_edit = QLineEdit()
        self.aux_arguments_line_edit = QLineEdit()
        self.log_data_path_line_edit.setText(os.getcwd())

        # Add the relevant widgets to the QGroupBox's layout
        opt_layout.addWidget(self.weight_table, 
                             0, 0, 1, 2)
        opt_layout.addWidget(QLabel('Regularization Coefficient:'),
                             1, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.reg_coeff_spinbox, 
                             1, 1, 1, 1)
        opt_layout.addWidget(QLabel('Avoid Bound Corners:'),
                             2, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.bounds_corners_check_box, 
                             2, 1, 1, 1)
        opt_layout.addWidget(QLabel('Log Data Path:'),
                             3, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.log_data_path_line_edit, 
                             3, 1, 1, 1)
        opt_layout.addWidget(QLabel('Prior Mean Model Path:'),
                             4, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.prior_mean_model_path_line_edit, 
                             4, 1, 1, 1)
        opt_layout.addWidget(QLabel('Prior Mean Auxiliary Arguments:'),
                             5, 0, 1, 1, alignment=Qt.AlignRight)
        opt_layout.addWidget(self.aux_arguments_line_edit, 
                             5, 1, 1, 1)
        
        # Create a right QWidget for proper resizing
        right_qwidget = QWidget()
        
        # Create a layout and add it to the sub QWidget
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_qwidget.setLayout(right_layout)
        
        # Create the QPushButton for initializing the GPBO model
        self.initialization_btn = QPushButton('Initialize and Iterate Once')

        # Add the relevant widgets to the right QWidget's layout
        right_layout.addWidget(acq_groupbox, 1)
        right_layout.addWidget(opt_groupbox, 3)
        right_layout.addWidget(self.initialization_btn)

        # add the relevant widgets to the main layout
        main_layout.addWidget(data_groupbox, 3)
        main_layout.addWidget(right_qwidget, 1)
        

class IterationPage(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create the main layout and add it to this page
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Create a QGroupBox to organize display widgets
        display_groupbox = QGroupBox('Display')
        
        # Create a layout and add it to the QGroupBox
        display_layout = QGridLayout()
        display_groupbox.setLayout(display_layout)
        
        # Create the QLabels for epoch number and the number of evaluation data
        self.epoch_label = QLabel('0')
        self.num_obj_data_label = QLabel('0')

        # Add the relevant widgets to the QGroupBox's layout
        display_layout.addWidget(QLabel('Iteration Epoch:'), 
                                 0, 0, 1, 1, alignment=Qt.AlignRight)
        display_layout.addWidget(self.epoch_label, 
                                 0, 1, 1, 1)
        display_layout.addWidget(QLabel('Number of Objective Evaluation Data:'), 
                                 1, 0, 1, 1, alignment=Qt.AlignRight)
        display_layout.addWidget(self.num_obj_data_label, 
                                 1, 1, 1, 1)

        # Create a QGroupBox to organize plot widgets
        plots_groupbox = QGroupBox('Preview Plots')
        
        # Create a layout and add it to the QGroupBox
        plots_layout = QVBoxLayout()
        plots_groupbox.setLayout(plots_layout)
        
        # Create the canvas for previewing plots
        self.canvas = FigureCanvas()

        # Add the relevant widgets to the QGroupBox's layout
        plots_layout.addWidget(self.canvas)

        # Create a QGroupBox to organize control widgets
        control_groupbox = QGroupBox('Control')
        
        # Create a layout and add it to the QGroupBox
        control_layout = QGridLayout()
        control_groupbox.setLayout(control_layout)
        
        # Create QSpinBoxes for selecting batch size and beta values
        self.batch_spin_box = QSpinBox()
        self.beta_spin_box = QDoubleSpinBox()
        self.plot_button = QCheckBox()
        
        # Set initial states
        self.batch_spin_box.setRange(1, 100)
        self.plot_button.setChecked(True)

        # Add the relevant widgets to the QGroupBox's layout
        control_layout.addWidget(QLabel('Batch Size:'), 
                                 0, 0, 1, 1, alignment=Qt.AlignRight)
        control_layout.addWidget(self.batch_spin_box, 
                                 0, 1, 1, 2)
        control_layout.addWidget(QLabel('Beta:'), 
                                 1, 0, 1, 1, alignment=Qt.AlignRight)
        control_layout.addWidget(self.beta_spin_box, 
                                 1, 1, 1, 2)
        control_layout.addWidget(QLabel('Draw Plot:'), 
                                 2, 0, 1, 1, alignment=Qt.AlignRight)
        control_layout.addWidget(self.plot_button, 
                                 2, 1, 1, 2)

        # Create a QGroupBox to organize data widgets
        data_groupbox = QGroupBox('Data')
        
        # Create a layout and add it to the QGroupBox
        data_layout = QGridLayout()
        data_groupbox.setLayout(data_layout)
        
        # Create the QTableWidgets for iteration parameters
        self.x_table = DecisionParameterTable(1, 3)
        self.y_table = ObjectiveTable(1)
        
        # Create the QPushButton for automatically calculating objective values
        self.obj_func_btn = QPushButton('Apply Objective Function')

        # Add the relevant widgets to the QGroupBox's layout
        data_layout.addWidget(QLabel('Decision Parameters'),
                              0, 0, 1, 1, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.x_table, 
                              1, 0, 2, 1)
        data_layout.addWidget(QLabel('Objective'), 
                              0, 1, 1, 1, alignment=Qt.AlignCenter)
        data_layout.addWidget(self.y_table, 
                              1, 1, 1, 1)
        data_layout.addWidget(self.obj_func_btn, 
                              2, 1, 1, 1)

        # Formatting layout shape
        data_layout.setColumnMinimumWidth(0, 600)
        data_layout.setColumnStretch(0, 1)

        # Create a QGroupBox to organize data widgets
        eval_pnt_groupbox = QGroupBox('Evaluation Point(s)')
        
        # Create a layout and add it to the QGroupBox
        eval_pnt_layout = QGridLayout()
        eval_pnt_groupbox.setLayout(eval_pnt_layout)
        
        # Create the QTableWidget for housing evaluation points
        self.eval_pnt_table = EvaluationPointTable(3)
        
        # Create the buttons for adding points to the decision parameter table
        self.add_points_button = QPushButton('Add All Points')
        self.add_point_button = QPushButton('Add Selected Point(s)')
        
        # Add the relevant widgets to the QGroupBox's layout 
        eval_pnt_layout.addWidget(self.eval_pnt_table, 0, 0, 1, 2)
        eval_pnt_layout.addWidget(self.add_points_button, 1, 0, 1, 1)
        eval_pnt_layout.addWidget(self.add_point_button, 1, 1, 1, 1)

        # Create the top-right QWidget for proper resizing
        top_right_qwidget = QWidget()
        
        # Create a layout and add it to the top-right QWidget
        top_right_layout = QHBoxLayout()
        top_right_layout.setContentsMargins(0, 0, 0, 0)
        top_right_qwidget.setLayout(top_right_layout)
        
        # Add the relevant widgets to the top-right QWidget's layout
        top_right_layout.addWidget(control_groupbox, 1)
        top_right_layout.addWidget(eval_pnt_groupbox, 3)

        # Create right and left QWidgets for proper resizing
        left_qwidget = QWidget()
        right_qwidget = QWidget()
        
        # Create layouts and add them to the right and left QWidgets
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setContentsMargins(0, 0, 0, 0)
        left_qwidget.setLayout(left_layout)
        right_qwidget.setLayout(right_layout)
        
        # Create the QPushButton for iterating the GPBO model
        self.iteration_button = QPushButton('Step Iteration')
        
        # Add the relevant widgets to the right and left QWidgets' layout
        left_layout.addWidget(display_groupbox, 1)
        left_layout.addWidget(plots_groupbox, 5)
        right_layout.addWidget(top_right_qwidget, 1)
        right_layout.addWidget(data_groupbox, 5)
        right_layout.addWidget(self.iteration_button, 1)

        # Add the relevant widgets to the main layout
        main_layout.addWidget(left_qwidget, 1)
        main_layout.addWidget(right_qwidget, 3)

class PlotsPage(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create the main layout and add it to this page
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Create the canvas for GPBO model plotting
        self.canvas = FigureCanvas()

        # Create the QRadioButtons for projection type
        self.acq_min_button = QRadioButton('Project Minimum')  # default
        self.acq_mean_button = QRadioButton('Project Mean')
        self.acq_fix_button = QRadioButton('Project By Fixing')

        # Set initial states
        self.acq_min_button.setChecked(True)
        
        # Create the QComboBoxes for axis selection
        self.acq_x_combobox = QComboBox()
        self.acq_y_combobox = QComboBox()
        
        # Create the QTableWidget for selecting fixed values
        self.acq_fixed_table = FixValueTable(1)
        
        # Create the QPushButton for applying updates of fixed projection values
        self.acq_fix_update_button = QPushButton('Update Fixed Dimensions')
        
        # Create a sub QWidget for proper sizing
        acq_xy_qwidget = QWidget()
        
        # Create a layout and add it to the sub QWidget
        acq_xy_layout = QGridLayout()
        acq_xy_qwidget.setLayout(acq_xy_layout)
        
        # Add the relevant widgets to the QWidget's layout
        acq_xy_layout.addWidget(QLabel('x-axis:'), 
                                0, 0, 1, 1, alignment=Qt.AlignRight)
        acq_xy_layout.addWidget(self.acq_x_combobox, 
                                0, 1, 1, 6)
        acq_xy_layout.addWidget(QLabel('y-axis:'), 
                                1, 0, 1, 1, alignment=Qt.AlignRight)
        acq_xy_layout.addWidget(self.acq_y_combobox, 
                                1, 1, 1, 6)

        # Create the QGroupBox for plot control regarding acquisition
        acq_groupbox = QGroupBox('Acquisition')
        
        # Create a layout and add it to the QGroupBox
        acq_layout = QVBoxLayout()
        acq_groupbox.setLayout(acq_layout)
        
        # Add the relevant widgets to the QGroupBox's layout
        acq_layout.addWidget(self.acq_min_button)
        acq_layout.addWidget(self.acq_mean_button)
        acq_layout.addWidget(self.acq_fix_button)
        acq_layout.addWidget(acq_xy_qwidget)
        acq_layout.addWidget(self.acq_fixed_table)
        acq_layout.addWidget(self.acq_fix_update_button)

        # Create the QRadioButtons for projection type
        self.post_min_button = QRadioButton('Project Minimum')  # default
        self.post_mean_button = QRadioButton('Project Mean')
        self.post_fix_button = QRadioButton('Project By Fixing')
        
        # Set initial states
        self.post_min_button.setChecked(True)

        # Create the QComboBoxes for axis selection
        self.post_x_combobox = QComboBox()
        self.post_y_combobox = QComboBox()
        
        # Create the QTableWidget for selecting fixed values
        self.post_fixed_table = FixValueTable(1)
        
        # Create the QPushButton for applying updates of fixed projection values
        self.post_fix_update_button = QPushButton('Update Fixed Dimensions')
        
        # Create a sub QWidget for proper sizing
        post_xy_qwidget = QWidget()
        
        # Create a layout and add it to the QWidget
        post_xy_layout = QGridLayout()
        post_xy_qwidget.setLayout(post_xy_layout)
        
        # Add the relevant widgets to the QWidget's layout
        post_xy_layout.addWidget(QLabel('x-axis:'), 
                                 0, 0, 1, 1, alignment=Qt.AlignRight)
        post_xy_layout.addWidget(self.post_x_combobox, 
                                 0, 1, 1, 6)
        post_xy_layout.addWidget(QLabel('y-axis:'), 
                                 1, 0, 1, 1, alignment=Qt.AlignRight)
        post_xy_layout.addWidget(self.post_y_combobox, 
                                 1, 1, 1, 6)
        
        # Create the QGroupBox for plot control regarding posterior mean
        post_groupbox = QGroupBox('Posterior Mean')
        
        # Create a layout and add it to the QGroupBox
        post_layout = QVBoxLayout()
        post_groupbox.setLayout(post_layout)
        
        # Add the relevant widgets to the QGroupBox's layout
        post_layout.addWidget(self.post_min_button)
        post_layout.addWidget(self.post_mean_button)
        post_layout.addWidget(self.post_fix_button)
        post_layout.addWidget(post_xy_qwidget)
        post_layout.addWidget(self.post_fixed_table)
        post_layout.addWidget(self.post_fix_update_button)

        # Create left and right QWidgets for proper resizing
        left_qwidget = QWidget()
        right_qwidget = QWidget()
        
        # Create left and right layouts and add them to their QWidget
        left_layout = QGridLayout()
        right_layout = QVBoxLayout()
        left_qwidget.setLayout(left_layout)
        right_qwidget.setLayout(right_layout)
        
        # Create Navigation Toolbar for main canvas
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, left_qwidget)

        # Create QSpinBox for plot history (epoch) selection
        # self.plot_spinbox = QSpinBox()
        
        # Add the relevant widgets to the left QWidget's layout
        left_layout.addWidget(self.nav_toolbar,
                              0, 0, 1, 1, alignment=Qt.AlignCenter)
        # left_layout.addWidget(self.plot_spinbox,
        #                       0, 1, 1, 1, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.canvas,
                              1, 0, 12, 2, alignment=Qt.AlignCenter)

        # Add the relevant widgets to the right QWidget's layout
        right_layout.addWidget(acq_groupbox)
        right_layout.addWidget(post_groupbox)

        # Add the left and right QWidgets to the main layout with proper sizing
        main_layout.addWidget(left_qwidget, 3)
        main_layout.addWidget(right_qwidget, 1)