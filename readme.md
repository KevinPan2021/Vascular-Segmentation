# Vessel Segmentation


### Build: 

	CPU: Intel Core Ultra 9 185H (16 cores)
	GPU: NVIDIA RTX 4090 (VRAM 16 GB)
	RAM: 32 GB
	Screen: 2560 x 1600 (150% Scale)



### Python Packages:

	conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
	conda install -c conda-forge tqdm = 4.66.5
	conda install -c conda-forge matplotlib = 3.9.2
	conda install -c conda-forge numpy = 1.26.4


### Code Structure:
```bash
├── VagSeg.py (Run to generate a GUI)
├── ssl_0.999alpha_multi_resolution_model.pth
├── data_process.py
├── read_write.py
├── UI_import.py
├── UI_manual.py
├── UI_utility.py
├── VasSeg.py
├── Vessel_Extraction.py
├── QT_import.ui
├── QT_main.ui
├── QT_manual.ui

```
	
	
