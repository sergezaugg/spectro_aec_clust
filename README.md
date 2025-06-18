# Compact auto-encoders for feature extraction from acoustic spectrograms  

### Table of Contents
* [Overview](#Overview)
* [Intallation](#Intallation)
* [Configuration](#Configuration)
* [Usage](#Usage)
* [ML details](#ML-details)

### Overview
* This is a codebase for applied research with auto-encoders to extract features from spectrograms 
* It allow two things:
* (1) Define and train simple custom Pytorch auto-encoders for spectrograms
* (2) Extract array features with these auto-encoders and convert them to linear features (details in pic below)
* Auto-encoders perform partial pooling of time axis (latent array representation is 2D -> channel by time)
* Specific data loader for spectrogram data to train under de-noising regime
* Extracted features are meant to be used in companion [project](https://github.com/sergezaugg/spectrogram_image_clustering) and its [frontend](https://spectrogram-image-clustering.streamlit.app/)

### Intallation  
* Developed under Python 3.12.8
* ```git clone https://github.com/sergezaugg/feature_extraction_saec```
* Make a fresh venv!
* Install basic packages with
* ```pip install -r requirements.txt```
* Ideally **torch** and **torchvision** should to be install for GPU usage
* This code was developed under Windows with CUDA 12.6 
* ```pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126```
* If other CUDA version or other OS, check official instructions [here](https://pytorch.org/get-started/locally)

### Configuration
* Paths where models will be saved and fetched are defined in **config/config.yaml**
* Parameters for multiple training sessions can be defined as json file in **./session_params/training**
* Parameters for multiple extraction sessions can be defined as yaml file in **./session_params/extraction**

### Usage 
* Prepare PNG formatted color images of spectrograms, e.g. with [this tool](https://github.com/sergezaugg/xeno_canto_organizer)
* [main_01_train.py](main_01_train.py) illustrates a pipeline to create and train auto-encoders
* [main_02_extract.py](main_02_extract.py) illustrates a pipeline to extract array features and get dim-reduced linear features
* Array and dim-reduced features are written to disk as NPZ files in parent of images dir.

### ML details
<img src="pics/flow_chart_01.png" alt="Example image" width="600"/>




