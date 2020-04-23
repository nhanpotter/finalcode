# Automated Scoring for Short Questions with Deep Learning

## Requirements

The code is written in [Python 3.7](https://www.python.org/), and it needs to be run inside a __virtual environment__ to isolate package installation from the system. Hence, Package manager, [pip](https://pip.pypa.io/en/stable/installing/), and [virtual environment](https://pypi.org/project/virtualenv/) is required in this project.

Please download [GloVe](https://www.kaggle.com/thanakomsn/glove6b300dtxt) before running the code.

### Libraries
* [TensorFlow 1.13.1](https://www.tensorflow.org/install/pip?lang=python3)
* [pandas](https://pandas.pydata.org/getting_started.html)
* [NumPy](https://numpy.org/)
* [Keras 2.1.0](https://keras.io/)
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [nltk](https://www.nltk.org/install.html)

_Note: tensorflow-gpu should be installed for GPU usage. For further information, click [here](https://www.tensorflow.org/install/gpu)_

## Deployment

### Regression model
```
python3 regression/main.py
```

### Classification model
```
python3 classification/main.py
```

### LSA approach
```
python3 lsa/main.py
```