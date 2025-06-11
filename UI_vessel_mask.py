# pyqt packages
from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QDialog, QLabel

# system packages
import os
import numpy as np
import cv2

# custom packages
from UI_utility import UI_Util

        
# Vessel Quant UI class
class UI_Vessel_Mask_Action(QDialog):
    label_quant_ReleaseSig = pyqtSignal()
    
    
    def __init__(self, filename, image, FOV, save_folder, parent_cls):
        super(UI_Vessel_Mask_Action, self).__init__()
        uic.loadUi('QT_vessel_mask.ui', self)
        self.setWindowIcon(QIcon('icons/logo.png'))
        self.setWindowTitle('Vessel Mask')
        
        self.parent_cls = parent_cls
        self._point = None
        
        # FOV should be in mm
        self.FOV = FOV
        self.Diameters = None
        if self.FOV == '3*3':
            self.FOV = 3
            self.Diameters = [int(x) for x in self.config.get('Default Parameters', 'diameter_3_3').split(',')]
        elif self.FOV == '6*6':
            self.FOV = 6
            self.Diameters = [int(x) for x in self.config.get('Default Parameters', 'diameter_6_6').split(',')]
        elif self.FOV == '12*12':
            self.FOV = 12
            self.Diameters = [int(x) for x in self.config.get('Default Parameters', 'diameter_12_12').split(',')]
        
        self.lineEdit_file.setText(filename)
        self.lineEdit_inner_diameter.setText(str(self.Diameters[0]))
        self.lineEdit_outer_diameter.setText(str(self.Diameters[1]))
        
        self.save_folder = save_folder
        
        self.original_enface = image
        
        self.load_mask()
        
        self.update_masked_enface_display()
        
        
    # If an attribute isn't found in this class, check in self.parent_cls.
    def __getattr__(self, name):
        return getattr(self.parent_cls, name)
        
    
    # overwrite te key press event
    def mouseReleaseEvent(self, event):
        widget = self.childAt(event.pos())
        
        if widget is None:
            return
        
        if isinstance(widget, QLabel) and widget.objectName==self.label_masked_enface.objectName:
            labelPos = self.label_masked_enface.mapFrom(self, event.pos())
            x, y = labelPos.x(), labelPos.y()
            self._point = QPointF(float(x/widget.width()),float(y/widget.height()))
            
            self.label_quant_ReleaseSig.emit()
            
        
    # link the qt commands trigger to actions
    def link_commands(self):
        # signals
        self.label_quant_ReleaseSig.connect(self.label_release_action)
        
        # buttons
        self.button_clear.clicked.connect(self.clear_action)
        self.button_finish.clicked.connect(self.finish_action)
        
        # line edits
        self.lineEdit_center.editingFinished.connect(self.center_action)
        self.lineEdit_inner_diameter.editingFinished.connect(self.inner_radius_action)
        self.lineEdit_outer_diameter.editingFinished.connect(self.outer_radius_action)
        
        
    # load mask (if exists)
    def load_mask(self):
        # check if the mask file exists
        filename_no_ext = os.path.splitext(os.path.basename(self.lineEdit_file.text()))[0]
        ring_path = f'{self.save_folder}/{filename_no_ext}/{os.path.basename(self.lineEdit_file.text())}_ring_mask.png'
        
        if not os.path.exists(ring_path):
            return
        
        ring_mask = cv2.imread(ring_path, cv2.IMREAD_GRAYSCALE)
        height, width = ring_mask.shape

        # Find contours with hierarchy info
        contours, hierarchy = cv2.findContours(ring_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        # Separate outer and inner contours using hierarchy
        radii_centers = []
        for idx, cnt in enumerate(contours):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            radii_centers.append((radius, (x, y)))
        
        # Sort to get inner and outer
        radii_centers.sort(key=lambda x: x[0])
        (inner_radius, center1), (outer_radius, center2) = radii_centers
        
        # Average centers for robustness
        center_x = (center1[0] + center2[0]) / 2
        center_y = (center1[1] + center2[1]) / 2

        # convert center and radius to mm
        center_x = round(center_x / height * self.FOV, 3)
        center_y = round(center_y / width * self.FOV, 3)
        inner_radius = round(inner_radius / height * self.FOV * 2, 3)
        outer_radius = round(outer_radius / width * self.FOV * 2, 3)
        
        # write to lineEdit
        self.lineEdit_center.setText(f'{center_x},{center_y}')
        self.lineEdit_inner_diameter.setText(str(inner_radius))
        self.lineEdit_outer_diameter.setText(str(outer_radius))
        
        # update display
        self.update_masked_enface_display()
        
    
    # mouse release to draw roi action
    def label_release_action(self):
        # extract the x and y coordinates
        x, y = float(self._point.x()), float(self._point.y()) 
        
        # convert it to mm
        x = round(x*self.FOV, 3)
        y = round(y*self.FOV, 3)
        
        # update text box
        self.lineEdit_center.setText(f'{x},{y}')
        
        # update display
        self.update_masked_enface_display()
        
    
    # clear button pressed action
    def clear_action(self):
        self.lineEdit_center.setText('')
        self.update_masked_enface_display()
        
        # also delete the saved mask file
        filename_no_ext = os.path.splitext(os.path.basename(self.lineEdit_file.text()))[0]
        save_folder = f'{self.save_folder}/{filename_no_ext}'
        
        save_path = os.path.join(save_folder, f"{os.path.basename(self.lineEdit_file.text())}_ring_mask.png")
        os.remove(save_path)
        
    
    # in mm unit
    def center_action(self):
        text = self.lineEdit_center.text()
        if not UI_Util.is_numeric(text):
            self.lineEdit_center.setText('')
        self.update_masked_enface_display()
        
        
    # in mm unit
    def inner_radius_action(self):
        text = self.lineEdit_inner_diameter.text()
        if not UI_Util.is_numeric(text):
            self.lineEdit_inner_diameter.setText(str(self.inner_radius_default))
        self.update_masked_enface_display()
        
        
    # in mm unit
    def outer_radius_action(self):
        text = self.lineEdit_outer_diameter.text()
        if not UI_Util.is_numeric(text):
            self.lineEdit_outer_diameter.setText(str(self.outer_radius_default))
        self.update_masked_enface_display()
                                                       
        
    
    # 
    def update_masked_enface_display(self):
        # Convert grayscale OpenCV image to RGB
        enface_rgb = cv2.cvtColor(self.original_enface, cv2.COLOR_GRAY2BGR)
        height, width, channel = enface_rgb.shape
        bytes_per_line = channel * width
    
        # Read inputs
        center_text = self.lineEdit_center.text()
        inner_radius_text = self.lineEdit_inner_diameter.text()
        outer_radius_text = self.lineEdit_outer_diameter.text()
        
        if not inner_radius_text:
            self.lineEdit_inner_diameter.setText(str(self.Diameter[0]))
        
        if not outer_radius_text:
            self.lineEdit_outer_diameter.setText(str(self.Diameter[1]))
            
        # Check all inputs are provided
        if center_text:
            # Parse center as "x,y"
            center_x, center_y = map(float, center_text.split(','))
            inner_radius = float(inner_radius_text)
            outer_radius = float(outer_radius_text)
            
            # convert from mm to pixels
            center_x = int(center_x / self.FOV * width)
            center_y = int(center_y / self.FOV * height)
            inner_radius = int(inner_radius / self.FOV * height / 2)
            outer_radius = int(outer_radius / self.FOV * height / 2)
            
            # Draw circle outlines (red) for inner and outer radius
            cv2.circle(enface_rgb, (center_x, center_y), inner_radius, (255, 0, 0), thickness=4)
            cv2.circle(enface_rgb, (center_x, center_y), outer_radius, (255, 0, 0), thickness=4)

        # Convert to QImage
        qimage = QImage(enface_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
        # Scale and display
        qpixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = qpixmap.scaled(
            self.label_masked_enface.width(),
            self.label_masked_enface.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label_masked_enface.setPixmap(scaled_pixmap)
        
    
    
    # finish button pressed, save ring mask and close the window
    def finish_action(self):
        # Read inputs
        center_text = self.lineEdit_center.text()
        inner_radius_text = self.lineEdit_inner_diameter.text()
        outer_radius_text = self.lineEdit_outer_diameter.text()
        
        if center_text and inner_radius_text and outer_radius_text:
            height, width = self.original_enface.shape
            # Parse center as "x,y"
            center_x, center_y = map(float, center_text.split(','))
            inner_radius = float(inner_radius_text)
            outer_radius = float(outer_radius_text)
            
            # convert from mm to pixels
            center_x = int(center_x / self.FOV * width)
            center_y = int(center_y / self.FOV * height)
            inner_radius = int(inner_radius / self.FOV * height / 2)
            outer_radius = int(outer_radius / self.FOV * height / 2)

            # Create blank mask
            mask = np.zeros((height, width), dtype=np.uint8)

            # Draw filled outer circle
            cv2.circle(mask, (center_x, center_y), outer_radius, 255, thickness=-1)
            # Subtract inner circle to create ring
            cv2.circle(mask, (center_x, center_y), inner_radius, 0, thickness=-1)
    
            # resize to (1024 * 1024)
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            
            # Save the binary ring mask
            filename_no_ext = os.path.splitext(os.path.basename(self.lineEdit_file.text()))[0]
            save_folder = f'{self.save_folder}/{filename_no_ext}'
            
            save_path = os.path.join(save_folder, f"{os.path.basename(self.lineEdit_file.text())}_ring_mask.png")
            cv2.imwrite(save_path, mask)

    
        # Close the pop-up window
        self.close()
    
    
        