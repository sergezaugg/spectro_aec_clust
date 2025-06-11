# Compact auto-encoders for feature extraction from acoustic spectrograms  

### Overview
* Define and train simple custom Pytorch auto-encoders for spectrograms
* Batch extract array features and linear features with these auto-encoders
* With partial pooling of time axis (latent representation is 2D -> channel by time)
* Specific data loader for spectrogram data to train under de-noising regime
* Extracted features are meant to be used in companion [project](https://github.com/sergezaugg/spectrogram_image_clustering) and its [frontend](https://spectrogram-image-clustering.streamlit.app/)

### Usage 
*  Prepare a naive autoencoder model with **main_00_make_models.py**
*  Prepare PNG formatted color images of spectrograms, e.g. with [this tool](https://github.com/sergezaugg/xeno_canto_organizer)
*  All functionality is called from 3 classes defined in **utils.py**
*  **main_01_train.py** illustrates a pipeline to train an auto-encoders with these images
*  Trained models are written to disk as PTH files 
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
* If other CUDA version or other OS, check official instructions [here](https://pytorch.org/get-started/locally)

### ML details

![](pics/flow_chart_01.png)


