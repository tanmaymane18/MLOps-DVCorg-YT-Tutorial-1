import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

FOLDS = 5
TEST_SIZE = 0.2

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

strfk = StratifiedKFold(n_splits=FOLDS)

for fold, (train_indices, val_indices) in enumerate(strfk.split(X=train.drop(columns=['DEATH_EVENT']), y=train['DEATH_EVENT'])):
    train.loc[val_indices, "Fold"] = fold

x_test = test.drop(columns=['DEATH_EVENT'])
y_test = test['DEATH_EVENT']

avg_auc = 0
avg_acc = 0
predictions = []

for fold in range(FOLDS):
    x_train = train[train["Fold"] != fold].drop(columns=['DEATH_EVENT', 'Fold'])
    x_val = train[train["Fold"] == fold].drop(columns=['DEATH_EVENT', 'Fold'])

    y_train = train[train["Fold"] != fold]['DEATH_EVENT']
    y_val = train[train["Fold"] == fold]['DEATH_EVENT']

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    clf = LogisticRegression()

    clf.fit(x_train, y_train)

    val_preds = clf.predict(x_val)
    test_preds = clf.predict_proba(x_test_scaled)

    predictions.append(test_preds)

    auc_score = roc_auc_score(y_val, val_preds)
    acc = accuracy_score(y_val, val_preds)

    print(f"{fold}: val_acc: {acc} val_auc: {auc_score}")

    with open('report.txt', 'a') as f:
        f.writelines(f"{fold}: val_acc: {acc} val_auc: {auc_score}\n")

    avg_auc += auc_score
    avg_acc += acc

test_preds = np.array(test_preds)
test_preds = np.mean(test_preds, axis=1)
test_preds = np.round(test_preds)

auc_score = roc_auc_score(y_test, test_preds)
acc = accuracy_score(y_test, test_preds)

with open('report.txt', 'a') as f:
        f.writelines(f"\n\ntest_acc: {acc} test_auc: {auc_score}\n")

c_mat = confusion_matrix(y_test, test_preds)

plt.figure(figsize=(10,8))
ax = sns.heatmap(c_mat, annot=True)
ax.set_title('Ensemble Confusion Matrix on holdout_data')
fig = ax.get_figure()
fig.savefig('confusion_mat.png')


