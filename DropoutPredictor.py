import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import csv

# Disable eager execution
tf.compat.v1.disable_eager_execution()
model = Sequential()
model.add(LSTM(64, input_shape=(3,1), return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(XXX, YYY, batch_size=32, epochs=20, verbose=1, validation_split=0.2)
predict_x=model.predict(X) 
classes_x = [1 if px[0] >= 0.5 else 0 for px in predict_x]

print("Accuracy Score -> ", accuracy_score(y_test, classes_x))
print("AUC Score -> ", roc_auc_score(y_test, classes_x))
print("Kappa Score -> ", cohen_kappa_score(y_test, classes_x))
print("F1 Score -> ", f1_score(y_test, classes_x))















