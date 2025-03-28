# Vessel Segmentation


### Build: 

	CPU: Intel i9-13900H (14 cores)
	GPU: NVIDIA RTX 4060 (VRAM 8 GB)
	RAM: 32 GB



### Python Packages:

	conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
	conda install -c conda-forge tqdm = 4.64.1
	conda install -c conda-forge matplotlib = 3.8.0
	conda install -c conda-forge numpy = 1.26.4



### Code Structure:
```bash
├── VagSeg.py (Run to generate a GUI)
├── model_adapted_train_anno_best.pth
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
	
	
