# pyqt packages
from PyQt5 import uic
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QDialog

# system packages
import numpy as np

# custom packages
from UI_utility import UI_Util

        
# Vessel Quant UI class
class UI_ColorBar_Action(QDialog):
    
    def __init__(self, img, metric_name, parent_cls):
        super(UI_ColorBar_Action, self).__init__()
        uic.loadUi('QT_colorbar.ui', self)
        self.setWindowIcon(QIcon('icons/logo.png'))
        self.setWindowTitle('Colorbar')
        self.parent_cls = parent_cls
        self.img = img # grayscale image
        self.metric_name = metric_name
        
        
        if self.metric_name == 'skeleMap_rg':
            minVal, maxVal = self.parent_cls.skeleMap_rg
        elif self.metric_name == 'diameter_rg':
            minVal, maxVal = self.parent_cls.diameter_rg
        elif self.metric_name == 'area_rg':
            minVal, maxVal = self.parent_cls.area_rg
        elif self.metric_name == 'complexity_rg':
            minVal, maxVal = self.parent_cls.complexity_rg
            
        self.lineEdit_contrast.setText(f'{minVal}-{maxVal}')
        
        
        # update the display
        self.update_display()
        
        
    # If an attribute isn't found in this class, check in self.parent_cls.
    def __getattr__(self, name):
        return getattr(self.parent_cls, name)
        
    
    def link_commands(self,):
        self.lineEdit_contrast.editingFinished.connect(self.Textbox_action)
        self.button_finish.clicked.connect(self.finish_action)
    
        
    # update label_colorbar display
    def update_display(self):
        # convert from gray to rgb
        # normalize to [0,1]
        minVal, maxVal = self.lineEdit_contrast.text().split('-')
        minVal, maxVal = float(minVal), float(maxVal)
        
        array = UI_Util.min_max_norm(self.img, (minVal, maxVal))
        
        # apply colormap
        colormap = UI_Util.dark_JET_cmap()
        img = colormap(array)[:,:,:3]*255 # convert rgba to rgb
        
        # Create a QImage from the NumPy array
        img = img.astype(np.uint8)
        h, w, _ = img.shape
        qimg = QImage(img.astype(np.uint8), w, h, w*3, QImage.Format_RGB888)
        
        # draw on pixmap
        qpixmap = QPixmap.fromImage(qimg)
        qpixmap = qpixmap.scaled(self.label_colorbar.width(), self.label_colorbar.height())
        self.label_colorbar.setPixmap(qpixmap)
        
    
        
    # changed text box action
    def Textbox_action(self):
        text = self.lineEdit_contrast.text().strip()
        valid, low, high = UI_Util.check_valid_contrast_input(text, 0, 100)
        if valid:
            self.update_display()
        else:
            self.lineEdit_contrast.setText(self.parent.prev_input)
        
    
    # finish button pressed, save ring mask and close the window
    def finish_action(self):
        minVal, maxVal = self.lineEdit_contrast.text().split('-')
        minVal, maxVal = float(minVal), float(maxVal)
        
        if self.metric_name == 'skeleMap_rg':
            self.parent_cls.skeleMap_rg = [minVal, maxVal]
        elif self.metric_name == 'diameter_rg':
            self.parent_cls.diameter_rg = [minVal, maxVal]
        elif self.metric_name == 'area_rg':
            self.parent_cls.area_rg = [minVal, maxVal]
        elif self.metric_name == 'complexity_rg':
            self.parent_cls.complexity_rg = [minVal, maxVal]
            
        
        self.parent_cls.update_quant_display()
        # Close the pop-up window
        self.close()
    
    
        