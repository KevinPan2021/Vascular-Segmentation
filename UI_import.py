# pyqt packages
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QColor, QPalette, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QDialog, QTableWidgetItem, QCheckBox, QWidget, QHBoxLayout, QLineEdit

# system packages
import os
import numpy as np

# custom package
from read_write import Read_Data
from data_process import Data


class UI_Select_Folder_Action(QDialog):
    
    def __init__(self, parent_cls):
        super(UI_Select_Folder_Action, self).__init__()
        uic.loadUi('QT_import.ui', self)
        self.setWindowIcon(QIcon('icons/UW.png'))
        self.setWindowTitle('Import Folder')
        
        self.parent_cls = parent_cls
        
        self.lineEdit_directory.setText(self.main_lineEdit_folder.text())
        
        # fill the tablewidget
        self.tableWidget.setColumnWidth(0, 355) # filename
        self.tableWidget.setColumnWidth(1, 80) # group
        self.tableWidget.setColumnWidth(2, 40)  # checkbox
        
        # collect all image files in the directory and group them
        self.files = dict()
        self._collect_files()
        
        # modify the tablewidget property
        self.tableWidget.setRowCount(len(self.files))
        palette = self.tableWidget.horizontalHeader().palette()
        palette.setColor(QPalette.Background, QColor("white"))
        self.tableWidget.horizontalHeader().setAutoFillBackground(True)
        self.tableWidget.horizontalHeader().setPalette(palette)
        
        # fill
        self._fill_tableWidget()
        
        
        
    # If an attribute isn't found in this class, check in self.parent_cls.
    def __getattr__(self, name):
        return getattr(self.parent_cls, name)
        
            
    # link the qt commands trigger to actions
    def link_commands(self):
        self.button_select_all.clicked.connect(self.select_all_action)
        self.button_deselect_all.clicked.connect(self.deselect_all_action)
        self.button_finish.clicked.connect(self.finish_action)
        
        self.tableWidget.itemSelectionChanged.connect(self.tableWidget_selection_action)
        
    
    # collect and group the files based on the filename
    def _collect_files(self):
        group_name = dict()
        for file in os.listdir(self.lineEdit_directory.text()):
            if file.endswith(('.png', '.jpg', '.tiff', '.tif', '.dcm', 'avi')):
                name = file.replace('enface','').replace('octa','').replace('oct','')
                if name not in group_name:
                    group_name[name] = len(group_name)
                self.files[file] = group_name[name]
        
        
        
    # fill the table widget cells       
    def _fill_tableWidget(self):
        
        # fill filenames and check box
        for i, (file, groupID) in enumerate(self.files.items()):
            
            # first column is the filename
            self.tableWidget.setItem(i, 0, QTableWidgetItem(file))
            
            # second column is the group name text edit
            line_edit = QLineEdit()
            line_edit.setText(str(groupID))
            self.tableWidget.setCellWidget(i, 1, line_edit)
        
            # third column is the checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # Set default to true
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.addWidget(checkbox)
            layout.setAlignment(checkbox, Qt.AlignCenter)  # Center the checkbox
            layout.setContentsMargins(0, 0, 0, 0)  # Remove extra margins
            self.tableWidget.setCellWidget(i, 2, widget)
                
            
    # selected a table cell, display the image
    def tableWidget_selection_action(self):
        # didn't select any file
        if len(self.tableWidget.selectedItems()) == 0:
            return
        
        item = self.tableWidget.selectedItems()[0]
        filename = f'{self.lineEdit_directory.text()}\{item.text()}'
        
        # read the image
        image = Read_Data(filename).get()
        
        # make sure the image is 2d
        if len(image.shape) == 3:
            image = np.max(image, axis=0)
        
        # Convert the OpenCV image to QImage
        height, width = image.shape
        qimage = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
    
        # Convert QImage to QPixmap and set it in the label
        qpixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = qpixmap.scaled(
            self.label_preview.width(),
            self.label_preview.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    
        # Set the scaled pixmap in the label
        self.label_preview.setPixmap(scaled_pixmap)
        
    
    # pressed select_all button action
    def select_all_action(self):
        # fill filenames and check box
        for i in range(self.tableWidget.rowCount()):
            checkbox = self.tableWidget.cellWidget(i, 2).findChild(QCheckBox)
            checkbox.setChecked(True)
        
        
    # pressed deselect_all button action
    def deselect_all_action(self):
        # fill filenames and check box
        for i in range(self.tableWidget.rowCount()):  
            checkbox = self.tableWidget.cellWidget(i, 2).findChild(QCheckBox)
            checkbox.setChecked(False)  # Check the checkbox
            
            
    # finish button pressed
    def finish_action(self):
        # check if valid
    
        # extract the selected image
        groupID_list = set()
        data = None
    
        # Get total checked rows for progress tracking
        total_checked = sum(
            1 for i in range(self.tableWidget.rowCount())
            if self.tableWidget.cellWidget(i, 2) and self.tableWidget.cellWidget(i, 2).findChild(QCheckBox).isChecked()
        )
    
        if total_checked == 0:
            return  # No files selected, exit early
        
        # initialize the progressbar
        self.progressBar.setMaximum(total_checked)
        self.progressBar.setValue(0)
    
        processed_count = 0
    
        for i in range(self.tableWidget.rowCount()):
            if self.tableWidget.item(i, 0) is None:
                break
    
            checkbox = self.tableWidget.cellWidget(i, 2).findChild(QCheckBox)
            if checkbox.isChecked():
                file = self.tableWidget.item(i, 0).text()
                groupID = self.tableWidget.cellWidget(i, 1).text()
    
                if not groupID in groupID_list:
                    if data is not None:
                        self.image_list.append(data)
                    data = Data(file)
                    groupID_list.add(groupID)
    
                img = Read_Data(os.path.join(self.lineEdit_directory.text(), file)).get()
    
                if 'enface' in file.lower():
                    data.enface = img
                elif 'octa' in file.lower():
                    data.OCTA = img
                elif 'oct' in file.lower():
                    data.OCT = img
    
                # Update progress bar
                processed_count += 1
                self.progressBar.setValue(processed_count)
                QApplication.processEvents()  # Ensures UI updates in real-time
        
        # add the last data
        self.image_list.append(data)
        
        # update the main displays
        self.fill_tableWidget_action()
    
        # close the pop-up window
        self.close()
    
    
        