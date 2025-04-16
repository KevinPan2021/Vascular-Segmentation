application_name = 'VasSeg V1.6'
import ctypes
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(application_name)

# pyqt packages
from PyQt5 import uic
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage, QPainter, QColor, QPalette
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QTableWidgetItem, QHeaderView

# system packages
import sys
import os
import cv2
import torch

# custom packages
from UI_utility import UI_Util
from UI_manual import UI_Manual_Action
from UI_import import UI_Select_Folder_Action
from UI_vessel_quant import UI_Vessel_Quant_Action
from read_write import Read_Data, Load_Model
from data_process import ProcessThread


device = "cuda" if torch.cuda.is_available() else "cpu"


class QT_Action(QMainWindow):

    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        uic.loadUi('QT_main.ui', self)

        self.setWindowIcon(QIcon('icons/UW.png')) # changed the window icon
        self.setWindowTitle(application_name) # set the title
        
        self.setWindowState(Qt.WindowActive)
        
        # runtime variables
        self.default_dir = 'C'
        self.processed = False
        self.selected_row = None
        self.output_folder = None
        self.dataset = None # image dataset class
        
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
        self.main_button_vessel_quant.clicked.connect(self.vessel_quant_action)
        
        # checkbox
        self.main_checkBox_model_output.toggled.connect(self._update_display)

        # tablewidgets
        self.main_tableWidget.itemSelectionChanged.connect(self.tableWidget_selection_action)
        
        
    
    # filling the tableWidget with file names
    def fill_tableWidget_action(self):
        if self.dataset is None:
            return

        self.main_tableWidget.setRowCount(len(self.dataset))
        
        for i in range(len(self.dataset)):
            # get the enface filepath
            self.main_tableWidget.setItem(i, 0, QTableWidgetItem(self.dataset.get_filepath(i)[2]))

        
    
    # selected a table cell, display the images
    def tableWidget_selection_action(self):
        # didn't select any file
        if len(self.main_tableWidget.selectedItems()) == 0:
            return
        
        self.selected_row = self.main_tableWidget.selectedItems()[0].row()
        self._update_display()
    
        
        
    # update the main display
    def _update_display(self):
        if self.selected_row is None:
            return
        
        # displaying the enface image
        enface_path = self.dataset.get_filepath(self.selected_row)[2]
        enface = Read_Data(enface_path).get()
        
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
        if self.output_folder is None:
            return
        
        enface_name = os.path.basename(enface_path)
        prediction_name = f'{enface_name}_prediction.png'
        model_output_name = f'{enface_name}_output.png'
        if not self.main_checkBox_model_output.isChecked() and prediction_name in os.listdir(self.output_folder):
            color_img = cv2.imread(f'{self.output_folder}/{prediction_name}', cv2.IMREAD_UNCHANGED)
            
            # Convert from RGBA to BGRA
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2RGBA)
            
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
            
            
        elif model_output_name in os.listdir(self.output_folder):
            color_img = cv2.imread(f'{self.output_folder}/{model_output_name}', cv2.IMREAD_COLOR)
            
            # Convert from RGB to BGR
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            height, width, channel = color_img.shape
    
            # Convert to QImage with RGBA format
            qimage = QImage(color_img.data, width, height, channel * width, QImage.Format_RGB888)
        
        else:
            return
            
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
        self.model = Load_Model(model_name, device).get()
        self.model.eval() # set the model into the evaluation mode
        
    
    # clicked select folder button action
    def select_folder_action(self):
        directory_path = QFileDialog.getExistingDirectory(None, "Select Directory", self.default_dir, options=QFileDialog.Options())
        
        # didn't select any files
        if directory_path is None or directory_path == '':
            return
        
        # create a output_folder
        self.output_folder = f'{directory_path}/VasSeg_output'
        os.makedirs(self.output_folder, exist_ok=True)
        
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
    
    
    def vessel_quant_action(self):
        '''
        if self.main_lineEdit_folder.text() == '':
            UI_Util.show_message(self, title='Action Error', message='Please import first')
            return
        '''
        new_window = UI_Vessel_Quant_Action(parent_cls = self)
        new_window.link_commands()
        new_window.exec_()
        
        
        
    def process_action(self):
        
        if self.main_lineEdit_folder.text() == '':
            self.button_process.setChecked(False)
            UI_Util.show_message(self, title='Action Error', message='Please load images first')
            return
            
        
        def run():
            
            # ask if really need to (re)process
            if not UI_Util.show_message_action(self, title='Run Check', message='Do you want to reprocess'):
                self.button_process.setChecked(False)
                return None
            
            
            self.processed = False
            self.button_process.setText('Stop')
            self.button_process.setChecked(True)

            self.thread = ProcessThread(self, device)
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
            
            if self.task is None:
                return
            
            self.task.process.connect(self.main_progressBar.setValue)  # Connect progress updates
            self.task.work_complete_signal.connect(finished)
            self.task.start()
        else:
            stopped()
    


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