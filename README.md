# hmm-engagement
Download with `git clone` or equivalent.
```bash
git clone @repo
```
* Python 3.x  
* Tensorflow > 1.5
* Sklearn > 0.19.0


## Installation

* initialise base classifier:
```python
pip install bayesian-hmm
pip install lime
```


## 1. Generating user representation: 
User representation needs to be generated so as to representing learner by her activities. The process depends on data structure, for big-data-edu representation see: ```big-data-edu```, for code yourself, see ```code-yourself```.


## 2. Model representation via HDP-HMM: 
For python2 see ```data-folder/HMM```, for python3, see ```HdpHmm.py```.

## 3. Dropout prediction & explainer: 
A sample implementation of LSTM and LIME explainer can be found in ```DropoutPredictor.py``` and ```Explainer.py``` respectively.
