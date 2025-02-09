# pyqt packages
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage, QFont, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QMessageBox


import configparser       
import numpy as np
import cv2
import math
import re



        
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

     