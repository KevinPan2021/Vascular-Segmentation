application_name = 'VasSeg V1.1'
import ctypes
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(application_name)

# pyqt packages
from PyQt5 import uic
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage, QPainter, QColor, QPalette
from PyQt5.QtCore import pyqtSignal, Qt, QPointF, QEvent, QSize, QThread
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QLineEdit, QDesktopWidget, QFileDialog, QCheckBox, QTableWidgetItem, QMessageBox

# system packages
import sys
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# custom packages
import model # DL model
from UI_utility import UI_Util
from UI_manual import UI_Manual_Action
from UI_import import UI_Select_Folder_Action
from read_write import Read_Data, Load_Model
from data_process import process_data, get_cavf_RGB, get_cavf_RGBA, get_cavf_Sparse_RGBA, ProcessThread


def setMouseTrackingRecursively(widget, enable):
    widget.setMouseTracking(enable)
    for child in widget.findChildren(QWidget):
        setMouseTrackingRecursively(child, enable)
        


class QT_Action(QMainWindow):

    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        uic.loadUi('QT_main.ui', self)

        self.setWindowIcon(QIcon('icons/UW.png')) # changed the window icon
        self.setWindowTitle(application_name) # set the title
        
        #QApplication.instance().focusChanged.connect(self.on_focus_changed)
        self.setWindowState(Qt.WindowActive)
        
        # runtime variables
        self.default_dir = 'C'
        self.processed = False
        self.selected_row = None
        self.image_list = [] # a list to keep all image_containers
        
        self._load_model()
        
        # modify the tablewidget property
        palette = self.main_tableWidget.horizontalHeader().palette()
        palette.setColor(QPalette.Background, QColor("white"))
        self.main_tableWidget.horizontalHeader().setAutoFillBackground(True)
        self.main_tableWidget.horizontalHeader().setPalette(palette)
        
        
    '''
    # window state changed
    def changeEvent(self, event):
        
        def resize_window(width, height):
            self.centralwidget.setFixedSize(QSize(width, height))
            
            self.label_image.setFixedSize(QSize(int(height*0.765), int(height*0.765)))
            self.label_mask.setFixedSize(QSize(int(height*0.765), int(height*0.765)))
            
            
        if event.type() == QEvent.WindowStateChange:
            # maximized window size
            if self.windowState() & Qt.WindowMaximized:
                screen = QDesktopWidget().screenGeometry()
                self.setMinimumSize(QSize(screen.width(), screen.height()))
                self.setMaximumSize(QSize(screen.width(), screen.height()))
                resize_window(screen.width(), int(screen.width()*0.58))
                
            # default window size  
            elif not (self.windowState() & Qt.WindowMinimized):
                screen = QDesktopWidget().screenGeometry()
                width = int(screen.width() * 0.85)
                height = int(width * 0.6)
                self.setMinimumSize(QSize(width, height))
                self.setMaximumSize(QSize(width, height))
                resize_window(width, height)
    '''
    
    # event link
    def link_commands(self):
        # tool buttons
        self.main_button_select_folder.clicked.connect(self.select_folder_action)
        self.button_process.clicked.connect(self.process_action)
        self.main_button_manual.clicked.connect(self.manual_correction_action)
        self.button_export.clicked.connect(self.export_action)
        
        # checkbox
        self.main_checkBox_model_output.toggled.connect(lambda: self.checkBox_model_output_action(True))
        
        # tablewidgets
        self.main_tableWidget.itemSelectionChanged.connect(self.tableWidget_selection_action)
        
        
    
    # filling the tableWidget with file names
    def fill_tableWidget_action(self):
        if not self.image_list:
            return
        
        self.main_tableWidget.setRowCount(len(self.image_list))
        for i in range(len(self.image_list)):
            self.main_tableWidget.setItem(i, 0, QTableWidgetItem(self.image_list[i].filename))
            
    
    # selected a table cell, display the images
    def tableWidget_selection_action(self):
        # didn't select any file
        if len(self.main_tableWidget.selectedItems()) == 0:
            return
        
        self.selected_row = self.main_tableWidget.selectedItems()[0].row()
        self._update_display()
    
    # 
    def checkBox_model_output_action(self, pressed=False):
        if not self.processed:
            self.main_checkBox_model_output.setChecked(False)
            return
        
        self._update_display()
        
        
    # update the main display
    def _update_display(self):
        if self.selected_row is None:
            return
        
        # displaying the enface image
        enface = self.image_list[self.selected_row].enface
        if enface is None:
            return
        
        # Convert the OpenCV image to QImage
        height, width = enface.shape
        gray_qimage = QImage(enface.data, width, height, width, QImage.Format_Grayscale8)
    
        # Convert QImage to QPixmap and set it in the label
        gray_pixmap = QPixmap.fromImage(gray_qimage)
        scaled_pixmap = gray_pixmap.scaled(
            self.main_label_enface.width(),
            self.main_label_enface.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    
        # Set the scaled pixmap in the label
        self.main_label_enface.setPixmap(scaled_pixmap)
        
        # displaying the prediction image
        if self.image_list[self.selected_row].prediction is None:
            return
        
        elif not self.main_checkBox_model_output.isChecked():
            color_img = self.image_list[self.selected_row].prediction
            # Get grayscale image dimensions
            gray_height, gray_width = enface.shape[:2] 
            
            # Convert color image to QImage (RGBA format)
            color_height, color_width, channel = color_img.shape
            color_qimage = QImage(color_img.data, color_width, color_height, channel * color_width, QImage.Format_RGBA8888)
            
            # Resize color image to match grayscale dimensions
            color_qimage = color_qimage.scaled(gray_width, gray_height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            
            # Convert QImage to QPixmap for overlaying
            color_pixmap = QPixmap.fromImage(color_qimage)
            
            # Create a new QPixmap to hold the blended image
            final_pixmap = QPixmap(gray_width, gray_height)
            final_pixmap.fill(Qt.transparent)  # Ensure transparency
            
            # Use QPainter to overlay images
            painter = QPainter(final_pixmap)
            painter.drawPixmap(0, 0, gray_pixmap)  # Draw grayscale image as background
            painter.drawPixmap(0, 0, color_pixmap)  # Overlay the resized RGBA image
            painter.end()
            
            # Convert back to QImage if needed
            qimage = final_pixmap.toImage()
            
        else:
            color_img = self.image_list[self.selected_row].model_output
            height, width, channel = color_img.shape
    
            # Convert to QImage with RGBA format
            qimage = QImage(color_img.data, width, height, channel * width, QImage.Format_RGB888)
            
            
        # Convert QImage to QPixmap and scale to fit label
        qpixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = qpixmap.scaled(
            self.main_label_prediction.width(),
            self.main_label_prediction.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    
        # Display the image in QLabel
        self.main_label_prediction.setPixmap(scaled_pixmap)
        
        
        
    
    def _load_model(self):
        model_name = self.main_comboBox_model.currentText()
        self.model = Load_Model(model_name, 'cuda').get()
        self.model.eval() # set the model into the evaluation mode
        
    
    # clicked select folder button action
    def select_folder_action(self):
        directory_path = QFileDialog.getExistingDirectory(None, "Select Directory", self.default_dir, options=QFileDialog.Options())
        
        # didn't select any files
        if directory_path is None or directory_path == '':
            return
        
        self.main_lineEdit_folder.setText(directory_path)
        
        new_window = UI_Select_Folder_Action(parent_cls = self)
        new_window.link_commands()
        new_window.exec_()    
        
        
    # manual correction button action
    def manual_correction_action(self):
        if self.main_lineEdit_folder.text() == '':
            UI_Util.show_message(self, title='Action Error', message='Please import first')
            return
        
        if self.selected_row is None:
            UI_Util.show_message(self, title='Action Error', message='Please Select an image first')
            return
        
        new_window = UI_Manual_Action(parent_cls = self)
        new_window.link_commands()
        new_window.exec_()
    
    
    # Disable gradient calculations
    @torch.no_grad()  
    def inference(self, data, proj_map):
        cavf_pred, ava_pred, cavf_pred_2D, ava_pred_2D = self.model(data, proj_map)
        cavf_pred_2D = F.softmax(cavf_pred_2D, dim=1)
        
        return cavf_pred_2D.squeeze().to('cpu').numpy()
    

    
    def process_action(self):
        def run():
            self.processed = False
            self.button_process.setText('Stop')
            self.button_process.setChecked(True)

            self.thread = ProcessThread(self)
            return self.thread
    
        def stopped():
            self.processed = False
            self.button_process.setText('Process')
            self.button_process.setChecked(False)
            if hasattr(self, 'thread') and self.thread.isRunning():
                self.thread.stop()  # Gracefully stop the process thread
                self.thread.quit()
                self.thread.wait()
    
        def finished():
            self.processed = True
            self.button_process.setText('Process')
            self.button_process.setChecked(False)
            self._update_display()
    
        # Toggle between Process/Stop
        if self.button_process.text() == 'Process':
            
            self.task = run()
            self.task.process.connect(self.main_progressBar.setValue)  # Connect progress updates
            self.task.work_complete_signal.connect(finished)
            self.task.start()
        else:
            stopped()
    

            
    # export the the model output, the label overlay, and the label
    def export_action(self):
        if not self.processed:
            UI_Util.show_message(self, title='Action Error', message='Please process first')
            return
        
        root = self.main_lineEdit_folder.text()
        
        for img in self.image_list:
            # BGR to RGB
            model_output = img.model_output[...,::-1]
            
            # Resize img.enface to match img.prediction's dimensions
            img_enface_resized = cv2.resize(img.enface, (img.prediction.shape[1], img.prediction.shape[0]))
            
            # Convert grayscale enface image to 3-channel (RGB)
            img_enface_rgb = cv2.cvtColor(img_enface_resized, cv2.COLOR_GRAY2RGB)
            
            # Extract RGBA channels from prediction
            pred_rgb = img.prediction[:, :, :3]  # RGB channels
            alpha = img.prediction[:, :, 3] / 255.0  # Normalize alpha to [0,1]
            
            # Blend the images using alpha blending
            overlayed = (1 - alpha[:, :, None]) * img_enface_rgb + alpha[:, :, None] * pred_rgb
            overlayed = overlayed.astype(np.uint8)  # Convert to uint8
            
            
            # Save the result
            output_folder = os.path.join(root, img.filename.rsplit('.', 1)[0])
            os.makedirs(output_folder, exist_ok=True)
            cv2.imwrite(os.path.join(output_folder, 'model_output.png'), model_output)
            cv2.imwrite(os.path.join(output_folder, 'mask.png'), img.prediction)
            cv2.imwrite(os.path.join(output_folder, 'overlayed_image.png'), overlayed)
        
        # pop up message
        if UI_Util.show_message_action(self, 'Done', 'Finished Exporting. Open Directory?', icon=QMessageBox.Information):
            os.startfile(root)
            

def main():
    app = QApplication(sys.argv)
    zoom = 100 / ctypes.windll.shcore.GetScaleFactorForDevice(0)
    
    app.setFont(QFont('微软雅黑', int(12*zoom)))
    
    app.setStyleSheet(f"""
        QToolButton {{ font-size: {8 * zoom}pt; }}
        QLabel {{ font-size: {12 * zoom}pt; }}
        QTabBar::tab {{ font-size: {10 * zoom}pt; }}
        QHeaderView::section {{ font-size: {8 * zoom}pt; }}
    """)

    action = QT_Action()
    action.show()
    action.link_commands()
    sys.exit(app.exec_())


if __name__ == "__main__":    
    main()