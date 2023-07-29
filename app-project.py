# Import necessary libraries


# Commented out IPython magic to ensure Python compatibility.
#Standard libraries for data analysis:

import numpy as np
import pickle
# import matplotlib.pyplot as plt
# import pandas as pd
# import pickle
# from scipy.stats import norm, skew
# from scipy.stats import ttest_ind


# # sklearn modules for data preprocessing

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# #sklearn modules for Model Selection

# from sklearn import svm, tree, linear_model, neighbors
# from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

# # import SMOTE module from Imbalanced Handling

# from imblearn.over_sampling import SMOTE
# from sklearn.pipeline import Pipeline, make_pipeline
# from imblearn.pipeline import Pipeline, make_pipeline


# #sklearn modules for Model Evaluation & Improvement

# from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, fbeta_score
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# from sklearn.model_selection import cross_val_score, GridSearchCV, ShuffleSplit, KFold

# from sklearn import feature_selection
# from sklearn import model_selection
# from sklearn import metrics

# from sklearn.metrics import classification_report, precision_recall_curve
# from sklearn.metrics import auc, roc_auc_score, roc_curve
# from sklearn.metrics import make_scorer, recall_score, log_loss
# from sklearn.metrics import average_precision_score


# #Standard libraries for data visualization

# import seaborn as sns
# from matplotlib import pyplot
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# import plotly.express as px
# import matplotlib
# color = sns.color_palette()
# import matplotlib.ticker as mtick
# from IPython.display import display
# pd.options.display.max_columns = None
# from pandas.plotting import scatter_matrix
# from sklearn.metrics import roc_curve


# #Miscellaneous Utilitiy Libraries

# import random
# import os
# import re
# import sys
# import timeit
# import string
# import time
# from datetime import datetime
# from time import time
# from dateutil.parser import parse
# import joblib

# """# Saving the Trained Model"""

# #filename = 'trained_model.sav' #name of saving file
# #pickle.dump(classifier, open(filename, 'wb')) #writting file with binary format of the model

#loading the saved model
loaded_model = pickle.load(open('D:/ZIDNA/IMPRTNT/Digital Skola/Final Project Data Science 26/Model Deployment/trained_model.sav', 'rb')) # Reading the binary format

input_data = (2, 0,	2,	3,	0,	5,	1,	0,	0,	0,	106.68,	1,	0,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1)

input_data_np = np.asarray(input_data)

input_data_reshape = input_data_np.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshape)
print(prediction)

if (prediction[0] == 0):
    print('The Booking order is cancelled')
else:
    print('The Booking order is not cancelled')