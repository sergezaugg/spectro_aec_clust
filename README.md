# Compact auto-encoders for feature extraction from acoustic spectrograms  

### Overview
* Allows to define simple custom Pytorch auto-encoders
* With partial pooling of time axis (latent representation is 2D -> channel by time)
* Has a specific data loader for spectrogram data
* Intended to experiment training under de-noising regime
* Some simple visuals to assess reconstruction 
* Trained AEC meant to be used in companion [projects](https://github.com/sergezaugg/spectrogram_image_clustering) and its [frontend](https://spectrogram-image-clustering.streamlit.app/)


### Dependencies / Intallation
* Developed under Python 3.12.8
* Make a fresh venv!
```bash 
pip install -r requirements.txt
```
* You also need to install **torch** and **torchvision**
* This code was developed under Windows with CUDA 12.6 
```bash 
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
* If other CUDA version needed, check instructions here https://pytorch.org/get-started/locally

### Usage 
*  see **main.py**


