import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from matplotlib import pyplot as plt

%matplotlib inline

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df.columns = [c.replace(' ', '_') for c in df.columns]
df['target'] = cancer.target
df['target'] = df.target.replace({0: 'malignant', 1: 'benign'})

# separation

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

# wrap in a function

def calculate_bacc(feature, percentile):
    threshold = np.percentile(X[feature], percentile)
    feature_cuts = np.where(X[feature] > threshold, 'left', 'right')
    decision = pd.DataFrame(zip(X[feature], feature_cuts, y), columns=['feature', 'cut', 'y'])
    majority = decision.groupby('cut')['y'].mean()
    # BUG: maybe could be the same, or maybe doesn't get rounded?
    left = round(majority['left'])
    right = round(majority['right'])
    decision['yhat'] = decision['cut'].replace({'left': left, 'right': right})
    bacc = balanced_accuracy_score(decision['y'], decision['yhat'])
    return bacc

# parametergrid

grid = ParameterGrid({
    'feature': X.columns,
    'percentile': [10, 20, 30, 40, 50, 60, 70, 80, 90]
})

fp = pd.DataFrame(grid)

fp['bacc'] = None
for i, row in fp.iterrows():
    bacc = calculate_bacc(row['feature'], row['percentile'])
    fp.at[i, 'bacc'] = bacc

fp = fp.sort_values('bacc', ascending=False)
feature, percentile, bacc = fp.head(1).values[0]

feature
percentile
bacc



from typing import List
from itertools import combinations

def gini(x: List[float]) -> float:
    x = np.array(x, dtype=np.float32)
    n = len(x)
    diffs = sum(abs(i - j) for i, j in combinations(x, r=2))
    return diffs / (2 * n**2 * x.mean())

X[feature]
threshold = np.percentile(X[feature], percentile)
feature_cuts = np.where(X[feature] > threshold, 'left', 'right')
decision = pd.DataFrame(zip(feature_cuts, y), columns=['cut', 'y'])
majority = decision.groupby('cut')['y'].mean()
left = int(round(majority['left']))
right = int(round(majority['right']))
decision['yhat'] = decision['cut'].replace({'left': left, 'right': right})

left = decision[decision['cut'] == 'left']['y'].values.tolist()
right = decision[decision['cut'] == 'right']['y'].values.tolist()

gini(left)
gini(right)

len(left)
len(right)

# then go to next level



###
