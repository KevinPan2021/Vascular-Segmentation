# pyqt packages
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QColor, QPen
from PyQt5.QtWidgets import QDialog, QLabel
from PyQt5.QtCore import QPoint, Qt

# system packages
import os
import cv2
import numpy as np
from skimage.morphology import remove_small_objects, label


# custom package
from UI_utility import UI_Util
from Vessel_Extraction import Vessel_Quantification
from read_write import Read_Data
from data_process import qpixmap_to_numpy, overlay



class DrawableLabel(QLabel):
    def initialize(self, parent):
        self.parent = parent  # Store reference to the main UI class
        
        # keep track of all actions (for undo)
        self.list_overlay_pixmap = []
        
        self.drawing = False
        self.last_point = QPoint()
        self.cursor_position = None  # Store cursor position for rendering
        
        self.setMouseTracking(True)  # Enable mouse tracking to trigger mouseMoveEvent without clicking
    
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Save previous state for undo before starting a new drawing action
            self.list_overlay_pixmap.append(self.parent.overlay_pixmap.copy())
            
            # can go back max_undo steps
            max_undo = 50
            while len(self.list_overlay_pixmap) >= max_undo:
                self.list_overlay_pixmap.pop(0)

            self.drawing = True
            self.last_point = event.pos()
            

    def mouseMoveEvent(self, event):
        if self.drawing and event.buttons() == Qt.LeftButton:
            painter = QPainter(self.parent.overlay_pixmap)

            # Determine pen color
            color_map = {
                'Vein': Qt.blue,
                'Artery': Qt.red,
                'FOVA': Qt.green,
                'None': Qt.transparent
            }
            color = color_map.get(self.parent.comboBox_type.currentText(), Qt.black)
            
            # If 'None', enable erasing mode
            if color == Qt.transparent:
                painter.setCompositionMode(QPainter.CompositionMode_Clear)

            # Set pen size
            size = int(self.parent.lineEdit_pen_size.text())
            painter.setPen(QPen(color, size, Qt.SolidLine))

            # Draw on the overlay
            painter.drawLine(self.last_point, event.pos())
            painter.end()

            self.last_point = event.pos()
            self.update()  # Refresh the QLabel to reflect changes

        # Store cursor position for rendering
        self.cursor_position = event.pos()
        self.update()  # Force UI refresh to show cursor preview

    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
        
            # Convert overlay to QImage
            overlay_image = self.parent.overlay_pixmap.toImage()
            binary_mask = self.parent.binary  # Get the latest binary mask

            # Convert binary numpy array to QImage
            height, width = binary_mask.shape
            binary_qimage = QImage(binary_mask.data, width, height, width, QImage.Format_Grayscale8)
            
            # Resize the binary mask to match overlay image dimensions
            binary_qimage = binary_qimage.scaled(
                overlay_image.size(),  # Match overlay size
                Qt.IgnoreAspectRatio,    # Ignore aspect ratio for perfect fit
                Qt.SmoothTransformation  # Apply smooth scaling
            )
        
            # Apply mask
            for y in range(overlay_image.height()):
                for x in range(overlay_image.width()):
                    # Get mask pixel value
                    mask_value = binary_qimage.pixelColor(x, y).red()  # Assuming grayscale
                    if mask_value == 0 and (overlay_image.pixelColor(x, y) == QColor(Qt.red) or overlay_image.pixelColor(x, y) == QColor(Qt.blue)):  # Dark regions in binary mask
                        overlay_image.setPixelColor(x, y, Qt.transparent)  # Make transparent
            
            # Convert back to QPixmap
            self.parent.overlay_pixmap = QPixmap.fromImage(overlay_image)
            
            self.update()  # Refresh QLabel
    
    
    # undo the previous step
    def undo_action(self):
        if not self.list_overlay_pixmap:
            UI_Util.show_message(self, title='Action Error', message='cannot go back further')
            return
        self.parent.overlay_pixmap = self.list_overlay_pixmap.pop()
        self.update()  # Refresh QLabel


    # Clears the overlay pixmap (removes all drawings)
    def clear_action(self):
        self.list_overlay_pixmap.append(self.parent.overlay_pixmap.copy())
        self.parent.overlay_pixmap.fill(Qt.transparent)  # Fill the pixmap with transparency
        self.update()  # Refresh QLabel

    
    # Removes small disconnected dots in overlay_pixmap and restores transparency
    def filter_action(self):
        # Convert overlay_pixmap to QImage
        overlay_image = self.parent.overlay_pixmap.toImage()
        width, height = overlay_image.width(), overlay_image.height()
    
        # Convert QImage to NumPy array (RGBA)
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                color = overlay_image.pixelColor(x, y)
                rgba_image[y, x] = [color.blue(), color.green(), color.red(), color.alpha()]
    
        # Extract only red and blue channels separately
        red_mask = (rgba_image[..., 0] == 0) & (rgba_image[..., 1] == 0) & (rgba_image[..., 2] > 0)
        blue_mask = (rgba_image[..., 0] > 0) & (rgba_image[..., 1] == 0) & (rgba_image[..., 2] == 0)
        green_mask = (rgba_image[..., 0] == 0) & (rgba_image[..., 1] > 0) & (rgba_image[..., 2] == 0)
        
        # Process red and blue separately
        min_size = int(self.parent.lineEdit_filter_area.text())
    
        labeled_red = label(red_mask)
        labeled_blue = label(blue_mask)
    
        filtered_red = remove_small_objects(labeled_red, min_size=min_size) > 0
        filtered_blue = remove_small_objects(labeled_blue, min_size=min_size) > 0
    
        # Reconstruct final RGBA image
        filtered_image = np.zeros_like(rgba_image)
        filtered_image[filtered_red] = [255, 0, 0, 255]  # Red
        filtered_image[filtered_blue] = [0, 0, 255, 255]  # Blue
        filtered_image[green_mask] = [0, 255, 0, 255] # Green
    
        # Convert NumPy array back to QImage
        final_qimage = QImage(filtered_image.data, width, height, width * 4, QImage.Format_RGBA8888)
    
        # Convert QImage to QPixmap
        self.parent.overlay_pixmap = QPixmap.fromImage(final_qimage)
    
        self.update()  # Refresh QLabel to reflect the filtered overlay

        
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        
        # Draw the overlay pixmap
        painter.drawPixmap(0, 0, self.parent.overlay_pixmap)

        # Draw the custom circular cursor without hiding the default cursor
        if self.cursor_position is not None:
            size = int(self.parent.lineEdit_pen_size.text())  # Get pen size
            color_map = {
                'Vein': Qt.blue,
                'Artery': Qt.red,
                'FOVA': Qt.green,
                'None': Qt.white
            }
            color = color_map.get(self.parent.comboBox_type.currentText(), Qt.black)

            pen = QPen(color, 2)  # Thin outline
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)  # Only draw an outline

            cursor_x = self.cursor_position.x()
            cursor_y = self.cursor_position.y()
            
            painter.drawEllipse(QPoint(cursor_x, cursor_y), size // 2, size // 2)  # Draw circle outline



            

class UI_Manual_Action(QDialog):
    
    def __init__(self, parent_cls = None):
        super(UI_Manual_Action, self).__init__()
        uic.loadUi('QT_manual.ui', self)
        self.setWindowIcon(QIcon('icons/UW.png'))
        self.setWindowTitle('Manual Segmentation')
        
        self.parent_cls = parent_cls
        
        # load the enface
        enface_fp = self.dataset.get_filepath(self.selected_row)[2]
        self.enface = Read_Data(enface_fp).get()
        
        # load the prediction
        self.enface_name = os.path.basename(enface_fp)
        prediction_name = f'{self.enface_name}_prediction.png'
        if prediction_name in os.listdir(self.parent_cls.output_folder):
            self.prediction = cv2.imread(f'{self.parent_cls.output_folder}/{prediction_name}', cv2.IMREAD_UNCHANGED)
            # Convert from RGBA to BGRA
            self.prediction = cv2.cvtColor(self.prediction, cv2.COLOR_BGRA2RGBA)
        else:
            self.prediction = None
            
        self.binary = None
        
        
        self.overlay_pixmap = QPixmap(self.label_overlay.size())  
        self.overlay_pixmap.fill(Qt.transparent)  # Transparent background
        
        # Promote `label_overlay` to DrawableLabel
        self.label_overlay.__class__ = DrawableLabel  # Change the class of the existing QLabel
        self.label_overlay.__class__.initialize(self.label_overlay, self)
       
        self.update_enface_display()
        
        
    # If an attribute isn't found in this class, check in self.parent_cls.
    def __getattr__(self, name):
        return getattr(self.parent_cls, name)
        

    def link_commands(self):
        self.button_binarize.clicked.connect(self.binarize_action)
        self.button_finish.clicked.connect(self.finish_action)
        self.button_fill_binary.clicked.connect(self.fill_binary_action)
        
        self.button_undo.clicked.connect(self.label_overlay.undo_action)
        self.button_filter.clicked.connect(self.label_overlay.filter_action)
        self.button_clearall.clicked.connect(self.label_overlay.clear_action)
        
    
    # update the label display
    def update_enface_display(self):
        # enface display
        if self.enface is None:
            return
        
        # Convert the OpenCV image to QImage
        height, width = self.enface.shape
        gray_qimage = QImage(self.enface.data, width, height, width, QImage.Format_Grayscale8)
    
        # Convert QImage to QPixmap and set it in the label
        gray_pixmap = QPixmap.fromImage(gray_qimage)
        scaled_pixmap = gray_pixmap.scaled(
            self.label_enface.width(),
            self.label_enface.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    
        # Set the scaled pixmap in the label
        self.label_enface.setPixmap(scaled_pixmap)
        
        
    
    def update_vessel_mask_display(self):
        # Convert the OpenCV image to QImage
        height, width = self.binary.shape
        binary = self.binary*255
        gray_qimage = QImage(binary.data, width, height, width, QImage.Format_Grayscale8)
    
        # Convert QImage to QPixmap and set it in the label
        gray_pixmap = QPixmap.fromImage(gray_qimage)
        scaled_pixmap = gray_pixmap.scaled(
            self.label_vessel_mask.width(),
            self.label_vessel_mask.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    
        # Set the scaled pixmap in the label
        self.label_vessel_mask.setPixmap(scaled_pixmap)
        
        
    def update_overlay_display(self):
        # Convert the OpenCV image to QImage
        height, width = self.enface.shape
        gray_qimage = QImage(self.enface.data, width, height, width, QImage.Format_Grayscale8)
    
        # Convert QImage to QPixmap and set it in the label
        gray_pixmap = QPixmap.fromImage(gray_qimage)
        scaled_pixmap = gray_pixmap.scaled(
            self.label_overlay.width(),
            self.label_overlay.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    
        # Set the scaled pixmap in the label
        self.label_overlay.setPixmap(scaled_pixmap)

        
    # binarize the enface image
    def binarize_action(self):
        thres = float(self.lineEdit_binary_threshold.text())
        self.binary = Vessel_Quantification(self.enface, thres).Ibinary2
        
        # processed
        if not self.prediction is None:
            # Convert to RGBA format for pixel manipulation
            color_image = self.prediction
            
            # Convert NumPy array to QImage
            height, width, channels = color_image.shape
            bytes_per_line = channels * width
            qimage = QImage(color_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
    
            # Resize cavf_pred_2D to match overlay_pixmap size
            qimage = qimage.scaled(self.overlay_pixmap.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        
            # Draw onto overlay_pixmap
            painter = QPainter(self.overlay_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)  # Preserve transparency
            painter.drawImage(0, 0, qimage)  # Overlay the processed image
            painter.end()

            
        # update the displays
        self.update_vessel_mask_display()
        self.update_overlay_display()
    
    
    def fill_binary_action(self):
        
        self.binary = np.ones_like(self.binary)
            
        # update the displays
        self.update_vessel_mask_display()
        self.update_overlay_display()
        
        
        
    # Rescale overlay_pixmap, convert to NumPy, and save it
    def finish_action(self):
        target_shape = self.enface.shape[:2]  # (height, width)
        
        # Resize overlay_pixmap to match cavf_pred_2D dimensions
        scaled_pixmap = self.overlay_pixmap.scaled(target_shape[1], target_shape[0], Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        # Convert the resized pixmap to a NumPy array
        prediction = qpixmap_to_numpy(scaled_pixmap)
        
        # Convert from RGBA to BGRA
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGRA2RGBA)
        
        overlayed = overlay(self.enface, prediction)
        
        cv2.imwrite(f'{self.parent_cls.output_folder}/{self.enface_name}_prediction.png', prediction)
        cv2.imwrite(f'{self.parent_cls.output_folder}/{self.enface_name}_overlay.png', overlayed)
        
        # Update display
        self.parent_cls._update_display()
        self.close()

        