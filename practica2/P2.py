# NAME: Aniol Juanola Vilalta (u1978893)
import pandas as pd
from graphviz import Graph


def read_model():
    df = pd.read_csv("insurance.csv")  # Loading the dataset into memory.
    # We should get rid of the instances which contain NA values:
    df = df.dropna()
    # since we cannot use strings as data to train the model, we have to give it a numeral label.
    from sklearn.preprocessing import LabelEncoder
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    le_paid = LabelEncoder()
    le_sex.fit(df["sex"])
    df["sex"] = le_sex.transform(df["sex"])
    le_smoker.fit(df["smoker"])
    df["smoker"] = le_smoker.transform(df["smoker"])
    le_region.fit(df["region"])
    df["region"] = le_region.transform(df["region"])
    df["paid"] = le_paid.fit_transform(df["paid"])  # equivalent
    return df


# 1) Load the insurance dataset to python. Prepare the dataset for correctly building the models.
# Justify all the steps and decisions you take. (0.5 point)
print("===================")
print("=   FIRST MODEL   =")
print("===================")

df = read_model()

# We can now divide our model into two parts: the training and testing part:
from sklearn.model_selection import train_test_split

x = df.drop(columns=["charges"])
y = df["charges"]

X_train, X_test, Y_Train, Y_test = train_test_split(x, y)

# 2) Build a decision tree for predicting the charge amount of the medical costs.
# Use all the variables in the dataset, leave all the tree parameters by default.
# Obtain the R2 and MAPE metrics with the train dataset and the test dataset.
# Is a good model? Why? Justify your answer (1 points)
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()

clf.fit(X_train, Y_Train)

y_pred = clf.predict(X_test)

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

print("EXERCISE 2 SCORE")
print("================")
print("R2 score: " + str(r2_score(y_pred, Y_test)))
print("MAPE score: " + str(mean_absolute_percentage_error(y_pred, Y_test)))
print()
# R2 score: 0.672829443235633
# MAPE score: 0.3916155811493374

# We can see that this model wouldn't be ideal, specially considering that the mean absolute percentage error is almost
# 40%. The R2 parameter isn't great either, it should be higher than 0.9

# 3) Delete the “patient” variable and set “max_depth” parameter to 4.
# Obtain the R2 and MAPE metrics with the train dataset and the test dataset again.
# Can you see any difference in the metrics obtained with the train and test datasets obtained previously?
# Why is this happening? Is this model better? (1.5 point) 

# Deleting the parameters and retesting:
df2 = df.drop(columns=["patient"])

x = df2.drop(columns=["charges"])
y = df2["charges"]

X_train, X_test, Y_Train, Y_test = train_test_split(x, y)

clf = DecisionTreeRegressor(max_depth=4)

clf.fit(X_train, Y_Train)

y_pred = clf.predict(X_test)
print("EXERCISE 3 SCORE")
print("================")
print("R2 score: " + str(r2_score(y_pred, Y_test)))
print("MAPE score: " + str(mean_absolute_percentage_error(y_pred, Y_test)))
print()
# R2 score: 0.8647331944757507
# MAPE score: 0.30560919679955606

# Both the R2 and MAPE score show better results than in the previous iteration. I wouldn't still consider it a good
# model, as the MAPE score is over 30% and R2 below 90%, but it's quite better than the previous one.

# 4) Construct the best model you can find for “charges”.
# Justify what you do and why you do each change and step (preprocess, variables, parameters, ….).
# Correct justifications and procedures will be more valuated than best scores. (2 points)
print("EXERCISE 4 SCORE")
print("================")

# In order to find the best model, we can look at the correlation between the different variables and "charges":
print(df.corr())
#           charges
# age        0.299008
# sex        0.057292
# bmi        0.198341
# children   0.067998
# smoker     0.787251
# region    -0.006208
# charges    1.000000
# patient   -0.003373
# paid      -0.408816

# It can be seen that the most useful variables will be "age", "bmi", "smoker" and "paid". We are going to use them:
df2 = df.drop(columns=["sex", "children", "region", "patient"])

x = df2.drop(columns=["charges"])
y = df2["charges"]

# In order to test the model, we will use cross-validation with KFold as it is a better way of finding the best model
# out of the data that we have. The chosen K for this example has been 10:

from sklearn.model_selection import KFold

k = 10
kf = KFold(n_splits=k)
model = DecisionTreeRegressor(max_depth=4,
                              criterion="absolute_error")  # TODO mirar els criterions que potser n'hi ha de més útils

r2_scores = []
mape_scores = []

for train_index, test_index in kf.split(x):
    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
    Y_Train, Y_test = y[train_index], y[test_index]

    model.fit(X_train, Y_Train)
    y_pred = model.predict(X_test)

    r2_scores.append(r2_score(y_pred, Y_test))
    mape_scores.append(mean_absolute_percentage_error(y_pred, Y_test))

avg_r2 = sum(r2_scores) / k
avg_mape = sum(mape_scores) / k

print("R2 scores: {}".format(r2_scores))
print("Avg R2 score: {}".format(avg_r2))
print("MAPE scores: {}".format(mape_scores))
print("Avg MAPE score: {}".format(avg_mape))

# The final model is as good as the one calculated before, according to the data from running the code on my laptop:
# Avg R2 score: 0.826895139728552
# Avg MAPE score: 0.3487349653902131

# On the 6th division, the models seems to work best:
# R2 score: 0.9165739590630678
# MAPE score. 0.18087626060005624

"""
Second model
"""
print()
print()
print("===================")
print("=   SECOND MODEL  =")
print("===================")

# 5) Reload the insurance dataset. Build a model for predicting if a bill will be paid or not.
# Do not delete any variable and leave the default configuration for the model.
# Obtain the confusion matrix and accuracy of the model for the test dataset. 
# Compute the precision and recall. (1 point)

df = read_model()

y = df["paid"]
x = df.drop(columns=["paid"])

X_train, X_test, Y_Train, Y_test = train_test_split(x, y)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X_train, Y_Train)

y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

print(classification_report(Y_test, y_pred))
print()
print(confusion_matrix(Y_test, y_pred))
print()
print("Accuracy score: " + str(accuracy_score(Y_test, y_pred)))
#               precision    recall  f1-score   support
#
#            0       0.87      0.59      0.70        22
#            1       0.97      0.99      0.98       313
#
#     accuracy                           0.97       335
#    macro avg       0.92      0.79      0.84       335
# weighted avg       0.96      0.97      0.96       335
#
#
# [False  True] (added for clarity)
# [[ 13   9]
#  [  2 311]]
#
# Accuracy score: 0.9671641791044776

# todo comentar

# 6) Do the same deleting the variable “patient” and "charges".
# Compare the results with the ones obtained before. These are good models?
# Correct explanations are more important than good codes. (2 point)  
df = df.drop(columns=["patient", "charges"])
y = df["paid"]
x = df.drop(columns=["paid"])

X_train, X_test, Y_Train, Y_test = train_test_split(x, y)

clf = DecisionTreeClassifier()

clf.fit(X_train, Y_Train)

y_pred = clf.predict(X_test)

print(classification_report(Y_test, y_pred))
print()
print(confusion_matrix(Y_test, y_pred))
print()
print("Accuracy score: " + str(accuracy_score(Y_test, y_pred)))
#               precision    recall  f1-score   support
#
#            0       0.72      0.59      0.65        22
#            1       0.97      0.98      0.98       313
#
#     accuracy                           0.96       335
#    macro avg       0.85      0.79      0.81       335
# weighted avg       0.96      0.96      0.96       335
#
#
# [False  True] (added for clarity)
# [[ 13   9]
#  [  5 308]]
#
# Accuracy score: 0.9582089552238806

# todo comentar

# 7) Construct the best model you can find for predicting “paid“ variable.
# Justify what you do and why you do each change and step  (preprocess, parameters, ….).
# Correct justifications and procedures will be more valuated than best scores.
# Visualize the tree you obtain. (2 points)
df = read_model()

# In order to train this model I will take a different approach.
# The variable correlation matrix is as useful as in the previous prediction, so we will repeat that but for the "paid"
# variable. This gives insight into which columns are useful for predicting the paid variable.
print(df.corr())
#               paid
# age      -0.206580
# sex      -0.027030
# bmi      -0.123426
# children -0.298265
# smoker   -0.270396
# region   -0.231221
# charges  -0.408816
# patient  -0.009834
# paid      1.000000

# The most relevant variables (20%+) seem to be "age", "children", "smoker", "region" and "charges". We will keep those.
df = df.drop(columns=["sex", "bmi", "patient"])
y = df["paid"]
x = df.drop(columns=["paid"])

# This time we will evaluate both the depth and criterion of the model, and choose the one which works best at the
# minimum depth given its accuracy.

# Once again, in order to have the same results all the time, we will use a KNN approach (to avoid the randomness of
# the train_test_split function).

from matplotlib import pyplot

k = 10
kf = KFold(n_splits=k)

for j in ["gini", "entropy", "log_loss"]:
    train_scores, test_scores = list(), list()
    for i in range(1, 20):
        model = DecisionTreeClassifier(max_depth=i,
                                       criterion=j)
        accuracy_scores_train = []
        accuracy_scores_test = []
        for train_index, test_index in kf.split(x):
            X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
            Y_train, Y_test = y[train_index], y[test_index]

            model.fit(X_train, Y_train)
            y_pred = model.predict(X_train)
            accuracy_scores_train.append(accuracy_score(y_pred, Y_train))

            y_pred = model.predict(X_test)
            accuracy_scores_test.append(accuracy_score(y_pred, Y_test))

        avg_acc_train = sum(accuracy_scores_train) / k
        avg_acc_test = sum(accuracy_scores_test) / k

        train_scores.append(avg_acc_train)
        test_scores.append(avg_acc_test)

    # plot the graph for each criterion
    pyplot.plot(range(1, 20), train_scores, "-o", label=str("Train_" + j))
    pyplot.plot(range(1, 20), test_scores, "-s", label=str("Test_" + j))
pyplot.legend()
pyplot.title("Analysis of the criterion and range for predicting the \"paid\" variable.")
pyplot.locator_params(axis='x', nbins=21)
pyplot.locator_params(axis='y', nbins=11)
pyplot.show()  # Must close plot window for the program to continue.

# As it can be seen (https://imgur.com/MLBEdSx) the optimal depth is 5 for the "gini" criterion while it is 6 for the
# remaining criteria. Further depth would imply that the model would be over-fitted for the given training data,
# so it would be less efficient really. The plot also suggests that "gini" is the best method as it approaches both
# the test data and train data best. So the best model would be the one fitted with the "gini" criterion at a level 5
# depth. Its average accuracy score is 0.975 for the test data and 0.985 for the training data. It's a success.

# Let's visualize the model itself:
model = DecisionTreeClassifier(max_depth=5,
                               criterion="gini")

# This time we will give it a random training approach by using the train_test_split function:
X_train, X_test, Y_Train, Y_test = (
    train_test_split(x, y, test_size=1 / k))  # maintaining the same proportion of training data

model.fit(X_train, Y_Train)

y_pred = model.predict(X_test)

# Importing the libraries and visualizing the graph
import graphviz
from sklearn.tree import export_graphviz

dot_data = export_graphviz(model, out_file=None,
                           class_names=["False", "True"], filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.view()
