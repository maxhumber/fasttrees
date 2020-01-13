import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from matplotlib import pyplot as plt

%matplotlib inline

cancer = load_breast_cancer()
cancer.keys()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df.columns = [c.replace(' ', '_') for c in df.columns]
df['target'] = cancer.target

df.info()
df.corrwith(df['target']).sort_values()

df['target'] = df.target.replace({0: 'malignant', 1: 'benign'})

target = 'target'
X = df.drop(target, axis=1)
y = df[target]

le = LabelEncoder()
y = le.fit_transform(y)

# experiment

feature = 'worst_concave_points'

p = 50
threshold = np.percentile(X[feature], 50)

feature_cuts = np.where(X[feature] > threshold, 'left', 'right')

decision = pd.DataFrame(zip(X[feature], feature_cuts, y), columns=['feature', 'cut', 'y'])
majority = decision.groupby('cut')['y'].mean()

# BUG: maybe could be the same, or maybe doesn't get rounded?
left = round(majority['left'])
right = round(majority['right'])

decision['yhat'] = decision['cut'].replace({'left': left, 'right': right})

bacc = balanced_accuracy_score(decision['y'], decision['yhat'])





###
