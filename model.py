# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from utils import from_str_to_list, normalize

# IMPORTING DATASET
dataset = pd.read_csv('dataset/jobs_data.csv', header = 0, index_col=False)

# DATA CLEANING
dataset.drop('Unnamed: 0', axis=1, inplace=True)
dataset['jobFunction'].replace('[\'nan\']', np.nan, inplace=True)
dataset.dropna(subset=['jobFunction'], inplace=True)

job_functions = [from_str_to_list(function) for function in dataset['jobFunction']]
job_functions_flattened = [item for sublist in job_functions for item in sublist]

unique_job_functions_flattened = set(job_functions_flattened)
classes = list(unique_job_functions_flattened)

title_words = [word for title in dataset['title'] for word in title.split()]
normalized_titles = [normalize(title) for title in dataset['title']]

# VECTORIZE YOUR DATA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3))
feature_matrix = vectorizer.fit_transform(normalized_titles)

multiLabelBinarizer = MultiLabelBinarizer()
labels = multiLabelBinarizer.fit_transform(job_functions)

# BUILD THE MODEL
from sklearn.svm import SVC 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.2)
estimator = SVC()
model = MultiOutputClassifier(estimator)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# EVALUEATE YOUR ALGORITHM
from sklearn.metrics import f1_score

score = f1_score(y_test, y_pred, average='weighted')
