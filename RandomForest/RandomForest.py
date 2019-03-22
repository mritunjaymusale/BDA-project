import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

dataset = pd.read_csv('../data/caravan-insurance-challenge.csv')
target = dataset['CARAVAN']
features = dataset.drop(['ORIGIN', 'CARAVAN'], axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(
    features, target, test_size=0.2, shuffle=True)

regressor = RandomForestClassifier(max_depth=6)


regressor.fit(xTrain,yTrain)
yPred = regressor.predict(xTest)

print(metrics.accuracy_score(yTest,yPred))