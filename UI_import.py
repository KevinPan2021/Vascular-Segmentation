# pyqt packages
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QColor, QPalette, QPixmap, QImage
from PyQt5.QtWidgets import QDialog, QTableWidgetItem, QCheckBox, QComboBox, QLineEdit, QFileDialog

# system packages
import os
import numpy as np

# custom package
from read_write import Read_Data, load_tsv
from data_process import Aireadi_Dataset
from UI_utility import UI_Util

  

class UI_Select_Folder_Action(QDialog):
    
    def __init__(self, parent_cls):
        super(UI_Select_Folder_Action, self).__init__()
        uic.loadUi('QT_import.ui', self)
        self.setWindowIcon(QIcon('icons/UW.png'))
        self.setWindowTitle('Import Folder')
        
        self.parent_cls = parent_cls
        self.tsv_dataset = None
        
        self.lineEdit_directory.setText(self.main_lineEdit_folder.text())
        
        # fill the tablewidget
        self.tableWidget.setColumnWidth(0, 525) # filename
        self.tableWidget.setColumnWidth(1, 65) # group
        self.tableWidget.setColumnWidth(2, 95) # data
        self.tableWidget.setColumnWidth(3, 40)  # checkbox
        
        # collect all image files in the directory and group them
        self.files = dict()
        self._collect_files()
        
        # modify the tablewidget property
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
        self.button_load_tsv.clicked.connect(self._load_tsv_action)
        
        self.tableWidget.itemSelectionChanged.connect(self.tableWidget_selection_action)
        
    
    # collect and group the files based on the filename
    def _collect_files(self):
        group_name = dict()
        for file in os.listdir(self.lineEdit_directory.text()):
            if file.endswith(('.png', '.jpg', '.tiff', '.tif', '.dcm', 'avi')):
                name = file.replace('enface','').replace('octa','').replace('oct','')
                if name not in group_name:
                    group_name[name] = len(group_name)
                
                if 'enface' in file:
                    self.files[f'{self.lineEdit_directory.text()}/{file}'] = ['enface', group_name[name]]
                elif 'octa' in file:
                    self.files[f'{self.lineEdit_directory.text()}/{file}'] = ['octa', group_name[name]]
                else:
                    self.files[f'{self.lineEdit_directory.text()}/{file}'] = ['oct', group_name[name]]
                    
    
    # clicked on the load tsv button, 
    def _load_tsv_action(self):
        file_path = QFileDialog.getOpenFileName(None, "Select a File", "", "All Files (*)")[0]
        
        self.files = dict()
        
        # didn't select any files
        if file_path is None or file_path == '':
            return
        
        # wrong format
        if not file_path.endswith('.tsv'):
            UI_Util.show_message(self, title='Action Error', message='File must be tsv')
            return
        
        self.tsv_dataset = load_tsv(file_path)
        
        #for group in range(len(self.tsv_dataset)):
        for group in range(10):
            oct_fp, octa_fp, enface_fp = self.tsv_dataset.get_filepath(group)
            self.files[enface_fp] = ['enface', group]
            self.files[oct_fp] = ['oct', group]
            self.files[octa_fp] = ['octa', group]
            
        # update the tablewidget
        self._fill_tableWidget()
        
    
    # filling the tableWidget based on self.files
    def _fill_tableWidget(self):
        self.tableWidget.setRowCount(len(self.files))
        
        # data is read from tsv, no need to group and set category
        if not self.tsv_dataset is None:
            for i, (file, [category, groupID]) in enumerate(self.files.items()):
                # First column - filename
                self.tableWidget.setItem(i, 0, QTableWidgetItem(file))
        
                # Second column - groupID as editable text
                self.tableWidget.setItem(i, 1, QTableWidgetItem(str(groupID)))
        
                # Third column - Use pre-created QComboBox
                self.tableWidget.setItem(i, 2, QTableWidgetItem(category))  # Use pre-created ComboBox
        
                # Fourth column - Checkbox
                checkbox_item = QTableWidgetItem()
                checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                checkbox_item.setCheckState(Qt.Checked)
                self.tableWidget.setItem(i, 3, checkbox_item)
                
        # no tsv file, need to group and set category 
        else:
            combo_options = ["enface", "oct", "octa"]
            combo_boxes = [QComboBox() for _ in range(len(self.files))]  # Pre-create ComboBoxes
            
            for combo_box in combo_boxes:  # Populate ComboBoxes before loop
                combo_box.addItems(combo_options)
                combo_box.setStyleSheet("color: white")  # Set style once
        
            for i, (file, [category, groupID]) in enumerate(self.files.items()):
                # First column - filename
                self.tableWidget.setItem(i, 0, QTableWidgetItem(file))
        
                # second column is the group name text edit
                line_edit = QLineEdit()
                line_edit.setText(str(groupID))
                self.tableWidget.setCellWidget(i, 1, line_edit)
        
                # Third column - Use pre-created QComboBox
                combo_box = combo_boxes[i]  # Get the corresponding QComboBox
                index = combo_box.findText(category)  # Find the index of the category
                if index != -1:
                    combo_box.setCurrentIndex(index)  # Set the default selection to category
                self.tableWidget.setCellWidget(i, 2, combo_box)  # Use pre-created ComboBox
        
                # Fourth column - Checkbox
                checkbox_item = QTableWidgetItem()
                checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                checkbox_item.setCheckState(Qt.Checked)
                self.tableWidget.setItem(i, 3, checkbox_item)


        
    # selected a table cell, display the image
    def tableWidget_selection_action(self):
        # didn't select any file
        if len(self.tableWidget.selectedItems()) == 0:
            return
        
        filename = self.tableWidget.selectedItems()[0].text()
        
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
            self.tableWidget.item(i, 3).setCheckState(Qt.Checked)
        
        
    # pressed deselect_all button action
    def deselect_all_action(self):
        # fill filenames and check box
        for i in range(self.tableWidget.rowCount()):  
            self.tableWidget.item(i, 3).setCheckState(Qt.Unchecked)
            
            
    # finish button pressed, save to data to main class and close the window
    def finish_action(self):
        # check if valid
    
    
        # load from tableWidget into data_list
        data_map = dict() # {group : {enface_path, oct_path, octa_path}}
        for i in range(self.tableWidget.rowCount()):
            if self.tableWidget.item(i, 0) is None:
                break
            
            # current row must be checked
            if self.tableWidget.item(i, 3).checkState():
                # choose the group, add new key in data_list for new group
                group = None
                if self.tsv_dataset:
                    group = self.tableWidget.item(i, 1).text()
                else:
                    group = self.tableWidget.cellWidget(i, 1).text()
                if not group in data_map:
                    data_map[group] = {'enface_path': None, 'oct_path': None, 'octa_path':None}
                
                # choose the category
                category = None
                if self.tsv_dataset:
                    category = self.tableWidget.item(i, 2).text()
                else:
                    category = self.tableWidget.cellWidget(i, 2).currentText()
                
                # add to data_list
                file_path = self.tableWidget.item(i, 0).text()
                if category == 'enface':
                    data_map[group]['enface_path'] = file_path
                elif category == 'oct':
                    data_map[group]['oct_path'] = file_path
                else:
                    data_map[group]['octa_path'] = file_path
        
        # load data_list into Aireadi_Dataset class
        self.parent_cls.dataset = Aireadi_Dataset(data_map, roi=550)
        
        # update the main displays
        self.parent_cls.fill_tableWidget_action()
    
        # close the pop-up window
        self.close()
    
    
        