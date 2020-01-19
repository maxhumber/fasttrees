from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import balanced_accuracy_score
%matplotlib inline

# from practical import PracticalTreeClassifier

# guidance
# https://cran.r-project.org/web/packages/FFTrees/vignettes/FFTrees_algorithm.html

# data prep

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df.columns = [c.replace(' ', '_') for c in df.columns]
df['target'] = cancer.target
df['target'] = df.target.replace({0: 1, 1: 0}) # reverse
df.to_csv('cancer.csv', index=False)
df = pd.read_csv('cancer.csv')

# train test split

target = 'target'
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# WIP build

feature = 'worst_concave_points'

max_levels = 4
max_thresholds = 10

thresholds = np.linspace(0, 100, num=max_thresholds+1)[1:-1]

percentile = thresholds[4]
threshold = np.percentile(X[feature], percentile)
directions = np.where(X[feature] <= threshold, 'left', 'right')

node = pd.DataFrame(zip(X[feature], directions, y), columns=['feature', 'direction', 'y'])
majority = node.groupby('direction')['y'].mean()

if majority['left'] <= majority['right']:
    left, right = 0, 1
else:
    left, right = 1, 0

node['yhat'] = node['direction'].replace({'left': left, 'right': right})

metric = balanced_accuracy_score

metric(node['y'], node['yhat'])

node

# package attempt 2

def score_feature(Xi, y, percentile=50, metric=balanced_accuracy_score):
    threshold = np.percentile(Xi, percentile)
    directions = np.where(Xi <= threshold, 'left', 'right')
    df = pd.DataFrame(zip(Xi, directions, y), columns=['feature', 'direction', 'y'])
    majority = df.groupby('direction')['y'].mean()
    if majority['left'] <= majority['right']:
        left, right = 0, 1
    else:
        left, right = 1, 0
    df['yhat'] = df['direction'].replace({'left': left, 'right': right})
    score = metric(df['y'], df['yhat'])
    return score, df

feature = 'worst_concave_points'
Xi = X[feature]
score, _ = score_feature(Xi, y)
_



### decide which direction to split next

from itertools import combinations

def gini(x):
    n = len(x)
    diffs = sum(abs(i - j) for i, j in combinations(x, r=2))
    return diffs / (2 * n**2 * x.mean())

gini_left = gini(node[node['direction'] == 'left']['y'])
gini_right = gini(node[node['direction'] == 'right']['y'])

if gini_left >= gini_right:
    next_split_direction = 'left'
else:
    next_split_direction = 'right'

X[node['direction'] == next_split_direction]




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
