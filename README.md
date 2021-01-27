# CLuster-based neArest neighbur based  IntRusion dEtection through convolutional neural network (CLAIRE)

The repository contains code refered to the work:


_Giuseppina Andresini, Annalisa Appice, Donato Malerba_

[Nearest cluster-based intrusion detection through convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S0950705121000617) 

Please cite our work if you find it useful for your research and work.
```
@article{ANDRESINI2021106798,
title = "Nearest cluster-based intrusion detection through convolutional neural networks",
journal = "Knowledge-Based Systems",
volume = "216",
pages = "106798",
year = "2021",
issn = "0950-7051",
doi = "https://doi.org/10.1016/j.knosys.2021.106798",
url = "http://www.sciencedirect.com/science/article/pii/S0950705121000617",
author = "Giuseppina Andresini and Annalisa Appice and Donato Malerba"
}

```

![alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0950705121000617-gr1.jpg)

## Code requirements

The code relies on the following **python3.6+** libs.

Packages need are:
* [Tensorflow 1.13](https://www.tensorflow.org/) 
* [Keras 2.3](https://github.com/keras-team/keras) 
* [Matplotlib 2.2](https://matplotlib.org/)
* [Pandas 0.23.4](https://pandas.pydata.org/)
* [Numpy 1.15.4](https://www.numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

## Data
The datasets used for experiments are accessible from [__DATASETS__](https://drive.google.com/open?id=1OIfsMv2PJljkc0aco00WB4_t8gEnXMiE). Original dataset is transformed in a binary classification: "_attack_, _normal_" (_oneCls files).
The repository contains the orginal dataset (folder: "original") and  the dataset after the preprocessing phase (folder: "numeric") 

Preprocessing phase is done mapping categorical feature and performing the Min Max scaler.

## How to use
* __main.py__ : script to run CLAIRE

 Tor run the code: main.py NameOfDataset (es CICIDS2017, UNSW__NB15 or KDDCUP99)
 
 
  

## Replicate the experiments

To replicate experiments reported in the work, you can use models and datasets stored in homonym folders.
Global variables are stored in __CLAIRE.conf__  file 


```python
    N_CLASSES = 2
    PREPROCESSING1 = 0  #if set to 1 code execute preprocessing phase on original date
    LOAD_AUTOENCODER = 1 #if 1 the autoencoder is loaded from models folder
    ORD_DATASET =1  #if 1 the dataset is ordered
    IMAGE=1 #if 1 the image dataset is loaded
    CLUSTERS=1 #if 1 the clustering step is performed
    LOAD_CLUSTERS=1 #if 1 the clustering model is loaded from models folder
    NUM_CLUSTERS=1000 #number of cluster trained
    LOAD_CNN = 1  #if 1 the classifier is loaded from models folder
    VALIDATION_SPLIT #the percentage of validation set used to train models
```

## Download datasets

[All datasets](https://drive.google.com/drive/folders/1OIfsMv2PJljkc0aco00WB4_t8gEnXMiE?usp=sharing)
