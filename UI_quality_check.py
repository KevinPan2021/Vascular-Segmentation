# pyqt packages
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QColor, QPen, QBrush
from PyQt5.QtWidgets import QDialog, QLabel
from PyQt5.QtCore import Qt, pyqtSignal

# system packages
import os
import cv2
import numpy as np
from matplotlib.path import Path


class UI_Quality_Check_Action(QDialog):
    label_ReleaseSig = pyqtSignal()
    
    def __init__(self, parent_cls = None):
        super(UI_Quality_Check_Action, self).__init__()
        uic.loadUi('QT_Quality_Check.ui', self)
        self.setWindowIcon(QIcon('icons/logo.png'))
        self.setWindowTitle('Quality Check')
        
        self.parent_cls = parent_cls
        
        # current file path
        self.filepath = os.path.basename(self.dataset.get_filepath(self.selected_row)[2])
        
        # background image with vessel overlays
        self.qpixmap = self.parent_cls.main_label_prediction.pixmap()
        
        # grid RGBA overlay
        grid_row, grid_coln = [int(x) for x in self.parent_cls.config.get('Default Parameters', 'quality_check_grid').split(',')]
        
        # good    -> 0 (green)
        # mid     -> 1 (yellow)
        # bad     -> 2 (red)
        # missing -> 3 (black)
        # line    -> 4 (white)
        self.grid = np.zeros((grid_row, grid_coln), dtype=np.uint16)
        self.grid_history = [self.grid.copy()]
        
        # current point clicked
        self.point = ()
        
        # draw region points
        self.region_points = []
        
        # select rectangle points
        self.rectangle_points = []
        
        # update the display to draw the qpixmap and grid overlay
        self.update_display()
        
        
    # If an attribute isn't found in this class, check in self.parent_cls.
    def __getattr__(self, name):
        return getattr(self.parent_cls, name)
        
    
    # link the commands to actions
    def link_commands(self):
        # button
        self.button_draw_region.clicked.connect(self.button_draw_action)
        self.button_select_rectangle.clicked.connect(self.button_rectangle_action)
        self.button_undo.clicked.connect(self.button_undo_action)
        self.button_finish.clicked.connect(self.finish_action)
        
        # checkbox
        self.checkBox_overlay.toggled.connect(self.update_display)
        
        # signal
        self.label_ReleaseSig.connect(self.label_clicked_action)
        
        
    # overwrite the mouse release event
    def mouseReleaseEvent(self, event):
        widget = self.childAt(event.pos())
        
        if widget is None:
            return
    
        # event on label_image
        if isinstance(widget, QLabel) and widget.objectName() == self.label_image.objectName():
            # Get position relative to label_image
            label_pos = self.label_image.mapFrom(self, event.pos())
    
            x, y = label_pos.x(), label_pos.y()
            label_width = self.label_image.width()
            label_height = self.label_image.height()
    
            # Normalize to [0, 1] range (fractional position)
            norm_x = x / label_width
            norm_y = y / label_height
            self.point = (norm_x, norm_y)
            self.label_ReleaseSig.emit()

            
        
    # clicked on the draw region button action
    def button_draw_action(self):
        self.button_select_rectangle.setChecked(False)
    
    
    # clicked on the select rectangle button action
    def button_rectangle_action(self):
        self.button_draw_region.setChecked(False)
        
        
    # update the label_image display
    def update_display(self):
        # Convert the OpenCV image to QImage
        scaled_pixmap = self.qpixmap.scaled(
            self.label_image.width(),
            self.label_image.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    
        if self.checkBox_overlay.isChecked():
            # Prepare the grid overlay as an RGBA image
            grid = self.grid  # Assuming shape is (H, W) like (32, 32)
            grid_rgba = np.zeros((grid.shape[0], grid.shape[1], 4), dtype=np.uint8)
            grid_rgba[grid == 0] = [0, 255, 0, 125]      # green
            grid_rgba[grid == 1] = [125, 125, 0, 125]    # yellow
            grid_rgba[grid == 2] = [255, 0, 0, 125]      # red
            grid_rgba[grid == 3] = [0, 0, 0, 125]        # black
            grid_rgba[grid == 4] = [255, 255, 255, 125]  # white
    
            # Convert grid to QImage
            grid_qimage = QImage(grid_rgba.data, grid.shape[1], grid.shape[0], grid.shape[1] * 4, QImage.Format_RGBA8888)
    
            # Scale grid overlay to match the display size
            scaled_grid_qimage = grid_qimage.scaled(
                scaled_pixmap.width(),
                scaled_pixmap.height(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation  # Nearest-neighbor
            )
    
            # Convert back to pixmap
            grid_pixmap = QPixmap.fromImage(scaled_grid_qimage)
    
            # Paint the grid overlay and dotted lines
            painter = QPainter(scaled_pixmap)
            painter.drawPixmap(0, 0, grid_pixmap)
    
            # Draw the full grid of dotted lines
            pen = QPen(QColor(255, 255, 255), 1, Qt.DotLine)
            painter.setPen(pen)
    
            num_rows, num_cols = grid.shape
            width = scaled_pixmap.width()
            height = scaled_pixmap.height()
            cell_width = width / num_cols
            cell_height = height / num_rows
    
            # Vertical lines
            for col in range(1, num_cols):
                x = int(col * cell_width)
                painter.drawLine(x, 0, x, height)
    
            # Horizontal lines
            for row in range(1, num_rows):
                y = int(row * cell_height)
                painter.drawLine(0, y, width, y)
            
            # Draw red dots for region and rectangle points
            dot_radius = 3
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 0, 0)))
    
            for point in self.region_points:
                if point is None:
                    continue
                x_norm, y_norm = point
                x_pix = int(x_norm * width)
                y_pix = int(y_norm * height)
                painter.drawEllipse(x_pix - dot_radius, y_pix - dot_radius, dot_radius * 2, dot_radius * 2)
            
            for point in self.rectangle_points:
                if point is None:
                    continue
                x_norm, y_norm = point
                x_pix = int(x_norm * width)
                y_pix = int(y_norm * height)
                painter.drawEllipse(x_pix - dot_radius, y_pix - dot_radius, dot_radius * 2, dot_radius * 2)
                
                
            painter.end()
    
        # Set the scaled pixmap in the label
        self.label_image.setPixmap(scaled_pixmap)


        
    # clicked on the label released action 
    def label_clicked_action(self):
        posX, posY = self.point
        
        # if draw_region button is selected
        if self.button_draw_region.isChecked():
            
            # first point
            if len(self.region_points) == 0:
                self.region_points.append((posX, posY))
            
            # none empty list
            else:
                first_point = self.region_points[0]
                distance = ((posX - first_point[0])**2 + (posY - first_point[1])**2) ** 0.5
                
                # should be enclosed
                if distance < 0.01:
                    self.region_points.append(None)
                    self.update_grid('draw_region')
                    
                # add to list
                else:
                    self.region_points.append((posX, posY))
            
            self.update_display()
        
        
        # if select_rectangle button is selected
        elif self.button_select_rectangle.isChecked():
            self.rectangle_points.append((posX, posY))
            
            # last point
            if len(self.rectangle_points) == 2:
                self.rectangle_points.append(None)
                self.update_grid('select_rectangle')
                
            self.update_display()
            
            
    # update the grid based on region or rectangle
    def update_grid(self, button):
        
        if self.comboBox_quality.currentText() == 'Good':
            quality = 0
        elif self.comboBox_quality.currentText() == 'Mid':
            quality = 1
        elif self.comboBox_quality.currentText() == 'Bad':
            quality = 2
        elif self.comboBox_quality.currentText() == 'Missing':
            quality = 3
        elif self.comboBox_quality.currentText() == 'Line':
            quality = 4
            
            
        n, m = self.grid.shape  # assuming square grid (n x n)
        
        if button == 'draw_region':
            # Close the region if needed
            region_points = [pt for pt in self.region_points if pt is not None]
            if not region_points:
                return  # nothing to do
            if region_points[0] != region_points[-1]:
                region_points.append(region_points[0])  # close the loop
    
            # Build polygon in (col, row) = (x, y) order for the Path mask
            polygon = [(x * n, y * m) for x, y in region_points]
            path = Path(polygon)
    
            # Generate grid of pixel centers
            xv, yv = np.meshgrid(np.arange(n), np.arange(m))
            points = np.vstack((xv.flatten(), yv.flatten())).T  # (x, y)
    
            mask = path.contains_points(points).reshape((n, m))
            self.grid[mask] = quality
            
            # Also mark individual clicked points explicitly
            for x, y in region_points:
                row = int(y * n)
                col = int(x * m)
                if 0 <= row < n and 0 <= col < m:
                    self.grid[row, col] = quality
            
            # empty out the region_point list
            self.region_points.clear()
            
            # add grid to history
            self.grid_history.append(self.grid.copy())
            
            
        elif button == 'select_rectangle':
            point1 = self.rectangle_points[0]
            point2 = self.rectangle_points[1]
            
            # extract upper left and lower right in [0,1]
            upper_left = (min(point1[0], point2[0]), min(point1[1], point2[1]))
            lower_right = (max(point1[0], point2[0]), max(point1[1], point2[1]))
            
            # clip to ensure within bounds
            row_start = np.clip(int(upper_left[1] * n), 0, n-1)
            row_end = np.clip(int(lower_right[1] * n), 0, n-1)
            col_start = np.clip(int(upper_left[0] * m), 0, m-1)
            col_end = np.clip(int(lower_right[0] * m), 0, m-1)
            
            # fill the grid in the rectangle
            self.grid[row_start:row_end+1, col_start:col_end+1] = quality
            
            # empty out the rectangle_points list
            self.rectangle_points.clear()
            
            # add grid to history
            self.grid_history.append(self.grid.copy())
            
        self.update_display()
        
    
    # undo button pressed action
    def button_undo_action(self):
        # set all buttons to unchecked
        self.button_draw_region.setChecked(False)
        self.button_select_rectangle.setChecked(False)
        
        # check if there are any previous grids in history
        if len(self.grid_history) == 0:
            return
        
        # empty out the region points and rectangle points
        self.region_points.clear()
        self.rectangle_points.clear()
        
        # replace the current grid with the last in grid history
        self.grid = self.grid_history.pop()

        # refresh the display
        self.update_display()
        
        
    # save and close the window
    def finish_action(self):
        grid = cv2.resize(self.grid, (256, 256), interpolation=cv2.INTER_NEAREST)
    
        filename_no_ext = os.path.splitext(os.path.basename(self.filepath))[0]
        output_folder = f'{self.parent_cls.output_folder}/{filename_no_ext}'
    
        # Helper to save mask
        def save_mask(path, mask):
            cv2.imwrite(path, (mask.astype(np.uint8)) * 255)
    
        save_mask(f'{output_folder}/quality_good.png', grid == 0)
        save_mask(f'{output_folder}/quality_mid.png', grid == 1)
        save_mask(f'{output_folder}/quality_bad.png', grid == 2)
        save_mask(f'{output_folder}/quality_missing.png', grid == 3)
        save_mask(f'{output_folder}/quality_line.png', grid == 4)
    
        self.close()
    
            
    