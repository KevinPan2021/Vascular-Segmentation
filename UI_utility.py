# pyqt packages
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage, QFont, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QMessageBox

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import configparser       
import re
import numpy as np
import scipy


        
class UI_Util():
    # pop up window
    def show_message(parent, title, message, icon=QMessageBox.Warning):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet(parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; min-height: 20px; color:white; \
                              background-color: rgb(91, 99, 120); border: 2px solid black; border-radius: 6px;}')
        msg_box.exec()
        
        
    # pop up window and return if the user selected "Yes"
    def show_message_action(parent, title, message, icon=QMessageBox.Question):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setStyleSheet(parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; min-height: 20px; color:white; \
                              background-color: rgb(91, 99, 120); border: 2px solid black; border-radius: 6px;}')
        return msg_box.exec() == QMessageBox.Yes
    
    
    # pop up window and return user's selection
    def show_message_selection(parent, title, message, choices, icon=QMessageBox.Question):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        # Add custom buttons (choices)
        for choice in choices:
            msg_box.addButton(choice, QMessageBox.AcceptRole)
        msg_box.setStandardButtons(QMessageBox.Close)
        msg_box.setStyleSheet(parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; min-height: 20px; color:white; \
                              background-color: rgb(91, 99, 120); border: 2px solid black; border-radius: 6px;}')
        return msg_box.exec()

    
    # Regular expression to match numbers (including integers and floats)
    def is_numeric(input_str):
        numeric_pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'
        return bool(re.match(numeric_pattern, input_str))
    
    
    # valid input format "a,b", where min<a<max, min<a<max and min<=a<=max, min<=a<=max
    # if valid, return "True, a,b". Else return "False, None, None"
    def check_valid_coord_input(text, minVal, maxVal):
        pattern = r'^\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*$'
        match = re.match(pattern, text)
        if match:
            a = float(match.group(1))
            b = float(match.group(3))
            if minVal <= a <= maxVal and minVal <= b <= maxVal:
                return True, a, b
        return False, None, None

        
    # valid input format "a-b", where a>=min, b<=max, and a < b
    def check_valid_contrast_input(text, minVal, maxVal):
        pattern = r'^\s*(-?\d+(\.\d+)?)\s*-\s*(-?\d+(\.\d+)?)\s*$'    
        match = re.match(pattern, text)
        if match:
            a = float(match.group(1))
            b = float(match.group(3))
            if a >= minVal and b <= maxVal and a < b:
                return True, a, b
        return False, None, None


    # valid input format "a", where a>min, a<max
    # if valid, return "True, a". Else return "False, None"
    def check_valid_float_input(text, minVal, maxVal):
        pattern = r'^-?\d+(\.\d+)?$'  # Pattern to match a float number
        if re.match(pattern, text):
            a = float(text)
            if minVal <= a <= maxVal:
                return True, a
        return False, None
    

    def min_max_norm(image, rg):
        image = np.clip(image, rg[0], rg[1])
        image = (image - rg[0]) / (rg[1] - rg[0])
        return image
    
    
    def resize_img(data, dimX, dimY):
        input_shape = data.shape
        
        # Calculate zoom factors for each dimension
        if len(input_shape) == 2:
            zoom_factors = (dimX / input_shape[0],  dimY / input_shape[1])
        elif len(input_shape) == 3:
            zoom_factors = (dimX / input_shape[0],  dimY / input_shape[1], 1)
        
        # Use scipy's zoom function to resize the image
        resized_data = scipy.ndimage.zoom(data, zoom_factors, order=1)
        
        # preserve_range
        data_min, data_max = data.min(), data.max()
        resized_data = np.clip(resized_data, data_min, data_max)
        
        return resized_data
    
    
    def dark_JET_cmap():
        jet = plt.colormaps['jet']
        colors = jet(np.linspace(0.15, 0.9, 256))
        colors = np.vstack((np.array([0, 0, 0, 1]), colors))
        return mcolors.LinearSegmentedColormap.from_list('modified_jet', colors)
    
    
    # takes an image and a colorbar pixmap, concatenate them
    def concatenate_colorbar_to_image(img):
        concatenated_image = QPixmap.fromImage(img)
        return concatenated_image
    
    
    # convert 2d np array to enlarged Qimage object
    def np_to_darkJET_Qimage(array, normRange, dim):
        # normalize to [0,1]
        array = UI_Util.min_max_norm(array, normRange)
        
        # apply colormap
        colormap = UI_Util.dark_JET_cmap()
        img = colormap(array)[:,:,:3]*255 # convert rgba to rgb
        
        # Create a QImage from the NumPy array
        img = UI_Util.resize_img(img, dim[0], dim[1]).astype(np.uint8)
        h, w, _ = img.shape
        qimage = QImage(img.astype(np.uint8), w, h, w*3, QImage.Format_RGB888)
        
        return qimage
    
    
     