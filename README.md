# ELEC0134_22-23 Assignment


## Description
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

## Requirements
numpy(1.22.3), pytorch(1.13.1), scikit-learn(1.0.2), opencv(4.6.0), dlib(19.24.0), tqdm(4.64.1)
