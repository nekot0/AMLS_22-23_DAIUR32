# ELEC0134_22-23 Assignment


## About this repository
This repository is part of the assignment of Applied Machine Learning Systems I (ELEC0134_22-23). "main.py" runs the models developed for each task - A1, A2, B1, and B2 - in a sequence. Hyperparameters are tuned so that each model outputs its most accurate results. To run "main.py", "Datasets" directory is needed on the same layer as A1, A2, B1, and B2 directory (see structure). 


## Structure
```
├── A1
│   ├── __init__.py
│   ├── A1.NN.py
│   ├── A1.SVM.py
│   ├── A1.AdaBoost.py
│   ├── load_celeba.py
│   └── shape_predictor_68_face_landmarks.dat
├── A2
│   ├── __init__.py
│   ├── A2.NN.py
│   ├── A2.SVM.py
│   └── A2.AdaBoost.py
├── B1
│   ├── __init__.py
│   ├── B1.SVM.py
│   ├── B1.NN.py
│   ├── load_cartoon.py
│   └── shape_predictor_81_face_landmarks.dat
├── B2
│   ├── __init__.py
│   ├── B2.SVM.py
│   └── B2.NN.py
├── Datasets (to be added manually)
│   ├── celeba
│   │   ├── img
│   │   │   └── 0-4999.jpg
│   │   └── labels.csv
│   ├── celeba_test
│   │   ├── img
│   │   │   └── 0-999.jpg
│   │   └── labels.csv
│   ├── cartoon_set
│   │   ├── img
│   │   │   └── 0-9999.png
│   │   └── labels.csv
│   └── cartoon_set_test
│       ├── img
│       │   └── 0-2499.png
│       └── labels.csv
│
├── main.py
└── README.md
```

## Descriptions
"main.py" calls "load_celeba.py" in A1 and "load_cartoon.py" in B1 and then runs models in A1, A2, B1, and B2 directory. 
- load_celeba.py: reads images from "celeba" and "celeba_test" directories and outputs (n, 68, 2)-dim matrix X and (n, 2)-dim matrix y, each of which contains 68 landmarks for images and the labels for task A1 and A2 (for task A1, y[i,0]=1 means i.jpg is labelled as male, and for task A2, as smiling). This file is used by models in A1 and A2 directories and requires "shape_predictor_68_face_landmarks.dat" file as a pre-trained data for the dlib face detector in the same directory.
- load_cartoon.py: reads images from "cartoon_set" and "cartoon_set_test" directories and outputs (n, 37)-dim matrix X and (n,5)-dim matrix y. X contains 19 features calculated using 81 landmarks and RGB colours sampled from 6 pixels representing skin and hair, and y contains labels (y[i,2]=1 means i.png is labelled as category 2). This file is used by models in B1 and B2 directories and requires "shape_predictor_81_face_landmarks.dat" file as a pre-trained data for the dlib face detector in the same directory.
- A1/NN.py, A1/SVM.py, A1/AdaBoost.py: each file contains the Neural Network, SVM, and AdaBoost models for task A1. They input 68 landmarks and the labels extracted by "load_celeba.py".
- A2/NN.py, A2/SVM.py, A2/AdaBoost.py: each file contains the Neural Network, SVM, and AdaBoost models for task A2. They input 68 landmarks and the labels extracted by "load_celeba.py".
- B1/SVM.py, B1/NN.py: each file contains the SVM and Neural Network models for task B1. They input 19 features and the labels extracted by "load_cartoon.py" and modified in "main.py".
- B2/SVM.py, B2/NN.py: each file contains the SVM and Neural Network models for task B2. They input 19 features and the labels extracted by "load_cartoon.py".

## Requirements
numpy(1.22.3), pytorch(1.13.1), scikit-learn(1.0.2), opencv(4.6.0), dlib(19.24.0), tqdm(4.64.1)
