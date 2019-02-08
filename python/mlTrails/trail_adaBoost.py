from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

# Intro
# https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

dataset = loadtxt('/Users/icce/Dropbox (Personal)/_thesis_framework/_scripts_hoy/r_icmi/python/mlTrails/pima-indians-diabetes.data.csv', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model to training data
model = AdaBoostClassifier()
model.fit(X_train, y_train)
print(model)



# predict feature importance
print(model.feature_importances_)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))