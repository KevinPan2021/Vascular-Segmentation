import numpy as np
import cv2
import torch
import pydicom
import pandas as pd
import os

# custom package
#from model import IPNV2, IPNV2_with_proj_map
from model_FiLM import IPNV2, IPNV2_with_proj_map


# load the model into device
class Load_Model():
    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.model = None
        

    
    def load_ssl(self, in_channels, n_classes, get_2D_pred, dc_norms, use_proj_map=True):
        if use_proj_map:
            cavf_model = IPNV2_with_proj_map(
                in_channels, n_classes, proj_map_in_channels=2, 
                get_2D_pred=get_2D_pred, dc_norms=dc_norms
            )
        else:
            cavf_model = IPNV2(in_channels, n_classes)
            
        state_dict = torch.load(
            'ssl_0.999alpha_multi_resolution_model.pth', 
            map_location=torch.device('cuda'), 
            weights_only=False
        )['teacher']
        cavf_model.load_state_dict(state_dict)
        return cavf_model
    
    
    
    # retrieve the model
    def get(self):
        if self.name == 'IPNV2':
            self.model = self.load_ssl(
                in_channels=2, n_classes=5, get_2D_pred=True, 
                dc_norms='NG', use_proj_map=True
            )
        self.model = self.model.to(self.device)
        
        return self.model
    


    
    
    

# read data into numpy array
class Read_Data():
    def __init__(self, filename):
        self.filename = filename
    
    
    # reading png or jpg as 2D numpy array
    def read_image(self):
        if self.filename.endswith('.png') or self.filename.endswith('.jpg'):
            image = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        return image
    
    
    # loads an .avi file and returns a 3D numpy array
    def load_avi(self):
        cap = cv2.VideoCapture(self.filename)

        slices = []
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if not np.all(frame[:, :, 0] == frame[:, :, 1]) or not np.all(frame[:, :, 1] == frame[:, :, 2]):
                raise ValueError("The input video is not grayscale")

            slices.append(frame[:, :, 0])

        cap.release()

        volume = np.stack(slices, axis=-1)
        return volume
    
    
    # loads an .dcm file and returns a 2D / 3D numpy array
    def load_dcm(self):
        data = pydicom.dcmread(self.filename).pixel_array
        if len(data.shape) == 3:
            data = data.transpose(1, 0, 2)
        return data

    
    # retrieve the image
    def get(self):
        if self.filename.endswith(('.png', '.jpg')):
            data = self.read_image()
        
        elif self.filename.endswith('.dcm'):
            data = self.load_dcm()
            
        elif self.filename.endswith('avi'):
            data = self.load_avi()
            
        return data
            


class load_tsv():
    def __init__(self, tsv_path):
        super().__init__()
        self.root = os.path.join(tsv_path, '..', '..')
        self.manifest = pd.read_csv(tsv_path, sep='\t')
        
    def __len__(self):
        return len(self.manifest)
    
    def get_filepath(self, idx):
        if idx >= len(self.manifest):
            raise IndexError
        
        row = self.manifest.iloc[idx]
        
        oct_fp = os.path.abspath(self.root + row['associated_structural_oct_file_path'])
        octa_fp = os.path.abspath(self.root + row['flow_cube_file_path'])
        enface_fp = os.path.abspath(self.root + row['associated_enface_1_file_path'])
        
        return oct_fp, octa_fp, enface_fp
    
    
    
    
    