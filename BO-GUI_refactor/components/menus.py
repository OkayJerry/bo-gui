import csv
import sys

from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import *

#############
# Functions #
#############

def parse_csv_to_dict(filename: str):
    # Open file and convert to a list
    with open(filename) as file:
        data = list(csv.reader(file, delimiter=","))
    
    # Initialize a dictionary
    d = {'xheader': [],
         'yheader': 'y0',
         'xdata': [],
         'ydata': [],
         'bounds': {'min': [],
                    'max': []}}

    # Fill the dictionary
    for element in data[0][:-1]:
        d['xheader'].append(element)
    d['yheader'] = data[0][-1]
    for row in data[1:-3]:
        d['xdata'].append(row[:-1])
        d['ydata'].append(row[-1])
    d['bounds']['min'] = data[-2][:-1]
    d['bounds']['max'] = data[-1][:-1]
    
    return d
    

class MenuBar(QMenuBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        parent = self.parent()
        
        # file menu
        new_action = QAction('&New', parent)
        open_action = QAction('&Open...', parent)
        save_action = QAction('&Save', parent)
        save_as_action = QAction('&Save As...', parent)
        exit_action = QAction('&Exit', parent)
        
        new_action.setShortcut(QKeySequence.New)
        open_action.setShortcut(QKeySequence.Open)
        save_action.setShortcut(QKeySequence.Save)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        exit_action.setShortcut(QKeySequence.Quit)

        # new_action.triggered.connect(self.newFile)
        open_action.triggered.connect(self.openFile)
        # save_action.triggered.connect(self.saveFile)
        # save_as_action.triggered.connect(self.saveFileAs)
        exit_action.triggered.connect(sys.exit)

        file_menu = self.addMenu('File')
        # file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        file_menu.addAction(exit_action)
        
    def openFile(self):
        while True:
            filename = QFileDialog.getOpenFileName(self.parent(), 'Open File', filter="All Files (*.*);;CSV File (*.csv);;PICKLE File (*.pickle)")[0]  # previously tuple
            if filename:
                if ".csv" not in filename and ".pickle" not in filename:
                    warning = QMessageBox()
                    warning.setIcon(QMessageBox.Critical)
                    warning.setText("Didn't select a *.csv or *.pickle file")
                    warning.setWindowTitle("ERROR")
                    warning.setStandardButtons(QMessageBox.Ok)

                    if warning.exec() == QMessageBox.Ok:
                        warning.close()
                        continue
            break

        if not filename:
            return
        
        elif ".csv" in filename:
            data = parse_csv_to_dict(filename)
            
            parent = self.parent()
            x_table = parent.initialization_page.x_table
            y_table = parent.initialization_page.y_table
            boundry_table = parent.initialization_page.boundry_table
            weight_table = parent.initialization_page.weight_table
            
            # Set X table
            x_table.setColumnCount(len(data['xheader']))
            x_table.setRowCount(len(data['xdata']))
            for i, header in enumerate(data['xheader']):
                x_table.setHorizontalHeaderItem(i, QTableWidgetItem(header))
            for i, row in enumerate(data['xdata']):
                for j, val in enumerate(row):
                    x_table.setItem(i, j, QTableWidgetItem(val))
    
            # Set Y table
            y_table.setRowCount(len(data['xdata']))
            y_table.horizontalHeaderItem(0).setText(data['yheader'])
            for i, val in enumerate(data['ydata']):
                y_table.setItem(i, 0, QTableWidgetItem(val))
                     
            # Set boundry table   
            boundry_table.setColumnCount(len(data['bounds']['min']))
            for i, header in enumerate(data['xheader']):
                boundry_table.setHorizontalHeaderItem(i, QTableWidgetItem(header))
            for i, val in enumerate(data['bounds']['min']):
                boundry_table.setItem(0, i, QTableWidgetItem(val))
            for i, val in enumerate(data['bounds']['max']):
                boundry_table.setItem(1, i, QTableWidgetItem(val))
                
            # Set weight table
            weight_table.setRowCount(len(data['xheader']))
            for i, header in enumerate(data['xheader']):
                weight_table.setItem(i, 0, QTableWidgetItem(1))
                weight_table.setVerticalHeaderItem(i, QTableWidgetItem(header))

        elif ".pickle" in filename:
            parent = self.parent()
            x_table = parent.iteration_page.x_table
            y_table = parent.iteration_page.y_table
            eval_pnt_table = parent.iteration_page.eval_pnt_table
            batch_spin_box =  parent.iteration_page.batch_spin_box
            beta_spin_box = parent.iteration_page.beta_spin_box
            tabs = parent.tabs
            epoch_label = parent.iteration_page.epoch_label
            
            x, y, bounds, batch_size, epoch, acquisition_type, acquisition_args = glb.interactive_GPBO.load_from_log(filename)

            # Fill arrays using X and Y data
            x_table.fillFromArray(x)
            y_table.fillFromArray(y)
            
            # Reset initialization table labels
            x_labels = [f'x0[{i}]' for i in range(iteration_x_table.columnCount())]
            x_table.setHorizontalHeaderLabels(x_labels)
            y_table.setHorizontalHeaderLabels(['y0'])
            
            # Adjust evaluation point table
            eval_pnt_table.setColumnCount(len(bounds))
            eval_pnt_table.setHorizontalHeaderLabels(x_labels)
            
            # Match batch size and epoch
            batch_spin_box.setValue(batch_size)
            epoch_label.setText(str(epoch))
            
            # Enable tabs for page navigation and move to iteration page
            tabs.setCurrentIndex(1)
            tabs.setTabEnabled(1, True)
            tabs.setTabEnabled(2, True)
            
            # Use acquisition type to enable/disable the 'beta' parameter
            if acquisition_type in ["UCB", "UpperConfidenceBound", "qUpperConfidenceBound", "LCB", "LowerConfidenceBound"]:
                beta_spin_box.setEnabled(True)
                beta_spin_box.setValue(acquisition_args['beta'])
            elif acquisition_type in ["KG", "qKnowledgeGradient", "KnowledgeGradient"]:
                beta_spin_box.setEnabled(False)
