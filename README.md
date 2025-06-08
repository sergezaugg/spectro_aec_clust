# Compact auto-encoders for feature extraction from acoustic spectrograms  

### Overview
* Allows to define and train simple custom Pytorch auto-encoders
* With partial pooling of time axis (latent representation is 2D -> channel by time)
* Has a specific data loader for spectrogram data
* Intended to experiment training under de-noising regime
* Some simple visuals to assess reconstruction 
* Trained AEC meant to be used in companion [projects](https://github.com/sergezaugg/spectrogram_image_clustering) and its [frontend](https://spectrogram-image-clustering.streamlit.app/)


### Usage quick-guide
*  Prepare a naive autoencoder model with **prepare_models/make_models.py**
*  Prepare PNG formatted color images of spectrograms, e.g. with [this tool](https://github.com/sergezaugg/xeno_canto_organizer)
*  All functionality is called from 3 classes defined in **utils.py**
*  **main_01_train.py** illustrates a pipeline to train an autoencoder with these images
*  Trained models are written to disk as pth files 
*  **main_02_extract.py** illustrates a pipeline to extract array features and get dim-reduced linear features
*  Array and dim-reduced features are written to disk as NPZ files


### Dependencies / Intallation
* Developed under Python 3.12.8
* Make a fresh venv!
* Install basic packages with
```bash 
pip install -r requirements.txt
```
* Ideally **torch** and **torchvision** should to be install for GPU usage
* This code was developed under Windows with CUDA 12.6 
```bash 
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
* If other CUDA version or other OS, check official instructions here https://pytorch.org/get-started/locally




