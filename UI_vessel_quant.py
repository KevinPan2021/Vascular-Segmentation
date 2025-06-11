# pyqt packages
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QDialog, QTableWidgetItem, QMessageBox

# system packages
import os
import numpy as np
import cv2
import csv
import pickle as pkl
from skimage.measure import regionprops, label

# custom package
from read_write import Read_Data
from UI_utility import UI_Util
from Vessel_Quant import Vessel_Quantification
from UI_vessel_mask import UI_Vessel_Mask_Action
from UI_colorbar import UI_ColorBar_Action


class ProcessingThread(QThread):
    progress_changed = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, tableWidget, comboBox_vessels, save_folder, skeleMap_rg, diameter_rg, area_rg, complexity_rg, png_dim):
        super().__init__()
        self.tableWidget = tableWidget
        self.comboBox_vessels = comboBox_vessels
        self.save_folder = save_folder
        self.png_dim = png_dim
        
        self.skeleMap_rg   = skeleMap_rg
        self.diameter_rg   = diameter_rg
        self.area_rg       = area_rg
        self.complexity_rg = complexity_rg
        
    
    # FAZ area, perimeter, and circularity quantification
    def FAZ_Quant(self, img, pixelsize):
        
        # Calculate area using skimage's area function (equivalent to bwarea)
        area = img.sum()  # Alternative to bwarea (skimage's area is just sum for binary masks)
        
        # Calculate perimeter using regionprops
        labeled_mask = label(img)
        stats = regionprops(labeled_mask)
        
        if not stats or stats[0].perimeter == 0:
            # Skip to next iteration (assuming inside a loop)
            return None, None, None
        
        perimeter_length = stats[0].perimeter
        
        # calculate FAZ area
        FAZarea = area * pixelsize ** 2  # in mm^2
        
        # calculate FAZ perimeter
        FAZperimeter = perimeter_length * pixelsize  # in mm
        
        # calculate FAZ circularity
        circularity = (4 * np.pi * area) / (perimeter_length ** 2)
        
        return FAZarea, FAZperimeter, circularity
    
    
    # AVA vein and artery quantification
    def AVA_Quant(self, img, pixelsize):
        vein_area = (img==0).sum() * pixelsize ** 2  # in mm^2
        artery_area = (img==2).sum() * pixelsize ** 2  # in mm^2
        
        return vein_area, artery_area
        
    
    # process the quantification based on the vessel type
    def process_vessel(self, img, file_name, vessel_name, img_size, ring_mask, FOV):
        filename_no_ext = os.path.splitext(os.path.basename(file_name))[0]
        save_folder = f'{self.save_folder}/{filename_no_ext}'
        
        # check if _manual exists
        suffix = '.png'
        if os.path.exists(f'{save_folder}/vein_mask_manual.png'):
            suffix = '_manual.png'

        if vessel_name == 'Vein':
            vessel_mask = cv2.imread(f'{save_folder}/vein_mask{suffix}', cv2.IMREAD_UNCHANGED)
        
        elif vessel_name == 'Artery':
            vessel_mask = cv2.imread(f'{save_folder}/artery_mask{suffix}', cv2.IMREAD_UNCHANGED)
        
        elif vessel_name == 'Capillary':
            vessel_mask = cv2.imread(f'{save_folder}/capillary_mask{suffix}', cv2.IMREAD_UNCHANGED)
        
        elif vessel_name == 'Large Vessels':
            vein_mask = cv2.imread(f'{save_folder}/vein_mask{suffix}', cv2.IMREAD_UNCHANGED)
            artery_mask = cv2.imread(f'{save_folder}/artery_mask{suffix}', cv2.IMREAD_UNCHANGED)
            vessel_mask = vein_mask | artery_mask
            
        elif vessel_name == 'All':
            vein_mask = cv2.imread(f'{save_folder}/vein_mask{suffix}', cv2.IMREAD_UNCHANGED)
            artery_mask = cv2.imread(f'{save_folder}/artery_mask{suffix}', cv2.IMREAD_UNCHANGED)
            capillary_mask = cv2.imread(f'{save_folder}/capillary_mask{suffix}', cv2.IMREAD_UNCHANGED)
            vessel_mask = vein_mask | artery_mask | capillary_mask
        
        # resize image
        img = cv2.resize(img, (img_size[0], img_size[1]), interpolation=cv2.INTER_NEAREST)
        if not ring_mask is None:
            ring_mask = cv2.resize(ring_mask, (img_size[0], img_size[1]), interpolation=cv2.INTER_NEAREST)
        vessel_mask = cv2.resize(vessel_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # convert to boolean
        vessel_mask = vessel_mask > 0 
        
        # extract the quantification results
        vessels = Vessel_Quantification(img, vessel_mask, ring_mask, FOV)

        skeleton = vessels.calc_skeleton()
        area_map, area_vals = vessels.calc_area_map()
        skeleton_map, skeleton_vals = vessels.calc_skeleton_map()
        diameter_map, diameter_vals = vessels.calc_diameter_map()
        _, _ = vessels.calc_perimeter_map()
        complexity_map, complexity_vals = vessels.calc_complexity_map()
        
        # save the raw images as pkl file
        raw = dict()
        raw['skeleton'] = skeleton
        raw['area_map'] = area_map
        raw['diameter_map'] = diameter_map
        raw['skeleton_map'] = skeleton_map
        raw['complexity_map'] = complexity_map
        with open(f'{save_folder}/{vessel_name}_maps.pkl', 'wb') as f:
            pkl.dump(raw, f)
        
        # save the rgb images
        skeleton = np.uint8(skeleton * 255)
        
        area_map = UI_Util.np_to_darkJET_Qimage(area_map, self.area_rg, self.png_dim)
        area_map = QPixmap.fromImage(area_map)
        
        diameter_map = UI_Util.np_to_darkJET_Qimage(diameter_map, self.diameter_rg, self.png_dim)
        diameter_map = QPixmap.fromImage(diameter_map)
        
        skeleton_map = UI_Util.np_to_darkJET_Qimage(skeleton_map, self.skeleMap_rg, self.png_dim)
        skeleton_map = QPixmap.fromImage(skeleton_map)
        
        complexity_map = UI_Util.np_to_darkJET_Qimage(complexity_map, self.complexity_rg, self.png_dim)
        complexity_map = QPixmap.fromImage(complexity_map)
        
        
        cv2.imwrite(f'{save_folder}/{vessel_name}_skeleton.png', skeleton)
        area_map.save(f'{save_folder}/{vessel_name}_area.png')
        diameter_map.save(f'{save_folder}/{vessel_name}_diameter.png')
        skeleton_map.save(f'{save_folder}/{vessel_name}_skeleton_map.png')
        complexity_map.save(f'{save_folder}/{vessel_name}_complexity.png')
        return area_vals, skeleton_vals, diameter_vals, complexity_vals
    
    
    def run(self):
        row_count = self.tableWidget.rowCount()
        for i in range(row_count):
            item = self.tableWidget.item(i, 0)
            if item is None:
                continue
            
            file_path = item.text()
            file_name = os.path.basename(file_path)
            
            # read the enface data
            img = Read_Data(file_path).get()
            
            FOV = self.tableWidget.item(i, 1).text()
            if FOV == '3*3':
                FOV = 3
                img_size = [420, 420]
                pixelsize = 3 / 420
            elif FOV == '6*6':
                FOV = 6
                img_size = [840, 840]
                pixelsize = 6 / 840
            elif FOV == '12*12':
                FOV = 12
                img_size = [1680, 1680]
                pixelsize = 12 / 1680
            
            # read ring mask (if exists)
            ring_mask = None
            
            filename_no_ext = os.path.splitext(os.path.basename(file_name))[0]
            save_folder = f'{self.save_folder}/{filename_no_ext}'
            
            ring_path = f'{save_folder}/ring_mask.png'
            if os.path.exists(ring_path):
                ring_mask = cv2.imread(ring_path, cv2.IMREAD_GRAYSCALE)
                
            # process vessels
            Vein_quant = self.process_vessel(img, file_name, 'Vein', img_size, ring_mask, FOV)
            Artery_quant = self.process_vessel(img, file_name, 'Artery', img_size, ring_mask, FOV)
            Capillary_quant = self.process_vessel(img, file_name, 'Capillary', img_size, ring_mask, FOV)
            Large_Vessels_quant = self.process_vessel(img, file_name, 'Large Vessels', img_size, ring_mask, FOV)
            All_quant = self.process_vessel(img, file_name, 'All', img_size, ring_mask, FOV)
            
            quant_lists = {'Vein':Vein_quant, 'Artery':Artery_quant, 'Capillary':Capillary_quant, 'Large_Vessels':Large_Vessels_quant, 'All':All_quant}
            
            # calculate FAZ variables
            for suffix in ['_manual.png', '.png']:
                FAZ_path = f'{save_folder}/FAZ_mask{suffix}'
                if os.path.exists(FAZ_path):
                    FAZ = cv2.imread(FAZ_path, cv2.IMREAD_UNCHANGED)
                    FAZ = (FAZ > 0).astype(np.uint8)
                    FAZ = cv2.resize(FAZ, (img_size[0], img_size[1]), interpolation=cv2.INTER_NEAREST)
                    FAZ_area, FAZ_perimeter, FAZ_circularity = self.FAZ_Quant(FAZ, pixelsize) 
                    break
            
            # calculate AVA variables
            for suffix in ['_manual.png', '.png']:
                AVA_path = f'{save_folder}/AVA_map{suffix}'
                if os.path.exists(AVA_path):
                    AVA = cv2.imread(AVA_path, cv2.IMREAD_UNCHANGED)
                    AVA = np.argmax(AVA, axis=-1).astype(np.uint8)
                    AVA = cv2.resize(AVA, (img_size[0], img_size[1]), interpolation=cv2.INTER_NEAREST)
                    
                    if not ring_mask is None:
                        ring_mask = cv2.resize(ring_mask, (AVA.shape[0], AVA.shape[1]), interpolation=cv2.INTER_NEAREST)
                        AVA = AVA * ring_mask
                        
                    AVA_vein_area, AVA_artery_area = self.AVA_Quant(AVA, pixelsize)
                    break
                
            # write to csv
            with open(f'{save_folder}/quantification.csv', mode='w', newline='') as f:
                writer = csv.writer(f)
                
                # write FAZ variables
                writer.writerow(['FAZ', 'values'])
                writer.writerow(['area (mm2)', FAZ_area])
                writer.writerow(['perimeter (mm)', FAZ_perimeter])
                writer.writerow(['circularity', FAZ_circularity])
                writer.writerow([])
                
                # write AVA variables
                writer.writerow(['AVA', 'values'])
                writer.writerow(['artery area (mm2)', AVA_artery_area])
                writer.writerow(['vein area (mm2)', AVA_vein_area])
                writer.writerow([])
                
                # write vessels variables
                for key, quant_list in quant_lists.items():
                    area_vals, skeleton_vals, diameter_vals, complexity_vals = quant_list
                    
                    writer.writerow([key, 'Inner', 'Ring', 'Outer', 'Total'])
                    writer.writerow(['Skeleton_Density'] + skeleton_vals)
                    writer.writerow(['Area'] + area_vals)
                    writer.writerow(['Diameter(mm)'] + diameter_vals)
                    writer.writerow(['Complexity'] + complexity_vals)
                    writer.writerow([])
                    
            # Update progress
            progress_percent = int((i + 1) / row_count * 100)
            self.progress_changed.emit(progress_percent)

        self.finished.emit()
        
        


# Vessel Quant UI class
class UI_Vessel_Quant_Action(QDialog):
    label_quant_MoveSig = pyqtSignal()
    
    def __init__(self, parent_cls, process_file=None):
        super(UI_Vessel_Quant_Action, self).__init__()
        uic.loadUi('QT_vessel_quant.ui', self)
        self.setWindowIcon(QIcon('icons/logo.png'))
        self.setWindowTitle('Vessel Quantification')
        
        self.parent_cls = parent_cls
        
        self.lineEdit_directory.setText(self.main_lineEdit_folder.text())
        
        self.process_file = process_file # either 
        self.selected_row = None # which file is selected for display
        self._point = None
        
        # set the display ranges
        self.skeleMap_rg   = [0, 0.2]
        self.diameter_rg   = [0, 0.5]
        self.area_rg       = [0, 0.1]
        self.complexity_rg = [0, 1  ]
        self.png_dim = [1024, 1024]
        
        
        # fill
        self._fill_tableWidget()
        
        
    # If an attribute isn't found in this class, check in self.parent_cls.
    def __getattr__(self, name):
        return getattr(self.parent_cls, name)
        
            
    # link the qt commands trigger to actions
    def link_commands(self):
        # signals
        self.label_quant_MoveSig.connect(self.mouse_move_action)
        
        # buttons
        self.button_process.clicked.connect(self.process_action)
        self.button_colorbar_range.clicked.connect(self.colorbar_range_action)
        self.button_mask.clicked.connect(self.mask_action)
        self.button_skeleton.clicked.connect(self.skeleton_action)
        self.button_skeleMap.clicked.connect(self.skeleMap_action)
        self.button_diameter.clicked.connect(self.diameter_action)
        self.button_area.clicked.connect(self.area_action)
        self.button_complexity.clicked.connect(self.complexity_action)
        self.button_finish.clicked.connect(self.finish_action)
        
        # tableWidget
        self.tableWidget.cellClicked.connect(self.on_cell_clicked)

        # comboBox
        self.comboBox_vessels.currentTextChanged.connect(self.update_quant_display)
        
    
    # filling the tableWidget based on according to main_tableWidget
    def _fill_tableWidget(self):
        
        # batch file
        if self.process_file is None:
            row_count = self.parent_cls.main_tableWidget.rowCount()
            self.tableWidget.setRowCount(row_count)
            
            for i in range(row_count):
                file_name = self.parent_cls.main_tableWidget.item(i, 0)
                FOV = self.parent_cls.main_tableWidget.item(i, 1)
                if file_name:
                    copied_item = QTableWidgetItem(file_name.text())
                    self.tableWidget.setItem(i, 0, copied_item)
                    
                    copied_item = QTableWidgetItem(FOV.text())
                    self.tableWidget.setItem(i, 1, copied_item)
        
        # single file
        else:
            self.tableWidget.setRowCount(1)
            
            file_name = self.parent_cls.main_tableWidget.item(self.process_file, 0)
            FOV = self.parent_cls.main_tableWidget.item(self.process_file, 1)
            if file_name:
                copied_item = QTableWidgetItem(file_name.text())
                self.tableWidget.setItem(0, 0, copied_item)
                
                copied_item = QTableWidgetItem(FOV.text())
                self.tableWidget.setItem(0, 1, copied_item)
                


    def on_cell_clicked(self, row, column):
        # Get the entire row as a list of strings
        self.selected_row = [
            self.tableWidget.item(row, col).text()
            for col in range(self.tableWidget.columnCount())
        ]
        self.update_enface_display()
        self.update_quant_display()
    
    
    # display the value at the mouse position on image
    def label_move_action(self,):
        img = None
        if self.button_skeleMap.isChecked():
            img = self.skeleton_map
        elif self.button_Diameter.isChecked():
            img = self.diameter_map
        elif self.button_Area.isChecked():
            img = self.area_map
        elif self.button_Complexity.isChecked():
            img = self.complexity_map
        
        # check if img is avaliable
        if img is None:
            return
        
        x, y = int(self._point.start.x()*self.OCT_Process_UI._nX), int(self._point.start.y()*self.OCT_Process_UI._nY)
        currVal = str(round(img[y,x], 2))        
        self.lineEdit_cursor.setText(currVal)
    
    
    # mask button pressed, pop up the mask window
    def mask_action(self):
        filename = self.selected_row[0]
        FOV = self.selected_row[1]
        
        # read the image
        image = Read_Data(filename).get()
        
        # make sure the image is 2d
        if len(image.shape) == 3:
            image = np.max(image, axis=0)
        
        new_window = UI_Vessel_Mask_Action(filename, image, FOV, self.parent_cls.output_folder, parent_cls = self)
        new_window.link_commands()
        new_window.exec_()
    
    
    # colorbar button clicked action
    def colorbar_range_action(self):
        # skeleton is grayscale
        if self.button_skeleton.isChecked():
            UI_Util.show_message(self, title='Action Error', message='skeleton has no colorbar')
            return
        
        file_path = self.selected_row[0]
        filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(os.path.basename(filename))[0]
        output_folder = f'{self.parent_cls.output_folder}/{filename_no_ext}'
        
        quant_name = self.comboBox_vessels.currentText() + '_maps.pkl'
        
        if not quant_name in os.listdir(output_folder):
            return
        
        with open(f'{output_folder}/{quant_name}', 'rb') as f:
            raw = pkl.load(f)
            
        
        if self.button_skeleMap.isChecked():
            image = raw['skeleton_map']
            new_window = UI_ColorBar_Action(image, 'skeleMap_rg', self)

        elif self.button_diameter.isChecked():
            image = raw['diameter_map']
            new_window = UI_ColorBar_Action(image, 'diameter_rg', self)

        elif self.button_area.isChecked():
            image = raw['area_map']
            new_window = UI_ColorBar_Action(image, 'area_rg', self)
            
        elif self.button_complexity.isChecked():
            image = raw['complexity_map']
            new_window = UI_ColorBar_Action(image, 'complexity_rg', self)
            
        new_window.link_commands()
        new_window.exec_()
        
        
        
    def update_enface_display(self):
        filename = self.selected_row[0]
        
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
            self.label_enface.width(),
            self.label_enface.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    
        # Set the scaled pixmap in the label
        self.label_enface.setPixmap(scaled_pixmap)
        
        
    # update the vessel 
    def update_quant_display(self):
        
        # displaying the prediction image
        if self.parent_cls.output_folder is None:
            return
        
        if self.selected_row is None:
            UI_Util.show_message(self, title='Action Error', message='please select a row first')
            return
        
        filename = self.selected_row[0]
        #enface_name = os.path.basename(filename)
        
        vessel_type = self.comboBox_vessels.currentText()
        
        quant_name = f'{vessel_type}_maps.pkl'
            
            
        filename_no_ext = os.path.splitext(os.path.basename(filename))[0]
        output_folder = f'{self.parent_cls.output_folder}/{filename_no_ext}'
        if quant_name in os.listdir(output_folder):
            
            with open(f'{output_folder}/{quant_name}', 'rb') as f:
                raw = pkl.load(f)
            
            
            if self.button_skeleMap.isChecked():
                image = raw['skeleton_map']
                qimage = UI_Util.np_to_darkJET_Qimage(image, self.skeleMap_rg, self.png_dim)
                
            elif self.button_diameter.isChecked():
                image = raw['diameter_map']
                qimage = UI_Util.np_to_darkJET_Qimage(image, self.diameter_rg, self.png_dim)
                
            elif self.button_area.isChecked():
                image = raw['area_map']
                qimage = UI_Util.np_to_darkJET_Qimage(image, self.area_rg, self.png_dim)
                
            elif self.button_complexity.isChecked():
                image = raw['complexity_map']
                qimage = UI_Util.np_to_darkJET_Qimage(image, self.complexity_rg, self.png_dim)
                
            # default to skeleton
            else:
                self.button_skeleton.setChecked(True)
                image = (raw['skeleton']*255).astype(np.uint8)
                height, width = image.shape
                qimage = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
                
            # Convert QImage to QPixmap and display
            qpixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = qpixmap.scaled(
                self.label_quant.width(),
                self.label_quant.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.label_quant.setPixmap(scaled_pixmap)
        
        '''
        # also update the lineEdit mean text
        # read from csv
        csv_path = ...
        if os.path.exists(csv_path):
            
            self.lineEdit_mean.setText()
        '''
        
    
    # update the line edits texts
    def mouse_move_action(self):
        ...
        '''
        # extract the x and y coordinates
        x, y = float(self._point.x()), float(self._point.y()) 
        
        # convert it to mm
        x = round(x*self.FOV, 3)
        y = round(y*self.FOV, 3)
        
        
        self.lineEdit_cursor.setText()
        '''
        
        
    # process button pressed action
    def process_action(self):

        # Set up processing thread
        self.thread = ProcessingThread(
            self.tableWidget, # tableWidget
            self.comboBox_vessels,
            self.parent_cls.output_folder, # output_folder
            self.skeleMap_rg,
            self.diameter_rg,
            self.area_rg,
            self.complexity_rg,
            self.png_dim
        )
    
        # Connect progress updates
        self.thread.progress_changed.connect(self.progressBar.setValue)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.start()
    
    
    def on_processing_finished(self):
        self.button_process.setChecked(False)
        
        # pop up window
        if UI_Util.show_message_action(self, 'Done', 'Finished Exporting. Open Directory?', icon=QMessageBox.Information):
            os.startfile(self.parent_cls.output_folder)
    
    
    # skeleton button pressed action
    def skeleton_action(self):
        # reset buttons
        self.button_skeleMap.setChecked(False)
        self.button_diameter.setChecked(False)
        self.button_area.setChecked(False)
        self.button_complexity.setChecked(False)
        
        # update display
        self.update_quant_display()
        
    
    # skeleMap button pressed action
    def skeleMap_action(self):
        # reset buttons
        self.button_skeleton.setChecked(False)
        self.button_diameter.setChecked(False)
        self.button_area.setChecked(False)
        self.button_complexity.setChecked(False)
        
        # update display
        self.update_quant_display()
        
        
    # diameter button pressed action
    def diameter_action(self):
        # reset buttons
        self.button_skeleton.setChecked(False)
        self.button_skeleMap.setChecked(False)
        self.button_area.setChecked(False)
        self.button_complexity.setChecked(False)
        
        # update display
        self.update_quant_display()
        
        
    # area button pressed action
    def area_action(self):
        # reset buttons
        self.button_skeleton.setChecked(False)
        self.button_skeleMap.setChecked(False)
        self.button_diameter.setChecked(False)
        self.button_complexity.setChecked(False)
        
        # update display
        self.update_quant_display()
        
        
    # complexity button pressed action
    def complexity_action(self):
        self.button_skeleton.setChecked(False)
        self.button_skeleMap.setChecked(False)
        self.button_diameter.setChecked(False)
        self.button_area.setChecked(False)
        
        # update display
        self.update_quant_display()
    
    
    
    # finish button pressed, save to data to main class and close the window
    def finish_action(self):
        # close the pop-up window
        self.close()
    
    
        