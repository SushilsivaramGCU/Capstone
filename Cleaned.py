#!/usr/bin/env python

__author__ = "Sushil Sivaram, Megha Gubbala", "Sylvia Nanyangwe"
__copyright__ = "N/A"
__credits__ = ["Isac Artzi", "Dinesh Sthapit", "Ken Ferrell", "James Dzikunu", "Tracy Roth", "Renee Morales"]
__license__ = "ECL"
__maintainer__ = "Sushil Sivaram, Megha Gubbala", "Sylvia Nanyangwe"
__email__ = "SushilSivaram@gmail.com"
__status__ = "Development"

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from dotenv import load_dotenv
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define Variables
load_dotenv()
#CSVData = os.getenv('CSVData')
CSVData = os.getenv('CSVOriginal')
DependentVariable = os.getenv('DependentVariable')
head_Value = int(os.getenv('Head_Value'))
testSize = float(os.getenv('test_size'))
randomstate = int(os.getenv('random_state'))

'''
Setup Reusable Functions
'''

# Load Data from CSV
def loadAndExtractData():
    global dataSetUp
    dataSetUp = pd.read_csv(CSVData)
    keepcolumns = ['cost_yr','median_income','affordability_ratio', 'ave_fam_size']
    dataSetUp =dataSetUp.filter(keepcolumns)
    for keep in keepcolumns:
        dataSetUp = dataSetUp[dataSetUp[keep].notna()]
    dataSetUp = dataSetUp[dataSetUp['median_income'] >= 8000]
    dataSetUp.to_csv('cleaned.csv', index=False)


# Print Info
def showDataHeadAndInfo(headCount):
    print(f"showing head {headCount} values")
    print(dataSetUp.head(headCount))
    print("**********")
    print("Showing info of dataset")
    print(dataSetUp.describe(include='all'))

# preProcessing
def preProcessing():
    histmedian_income =dataSetUp['median_income'].plot.hist(bins=5, alpha=0.5)
    hist_avg_fam =dataSetUp['ave_fam_size'].plot.hist(bins=5, alpha=0.5)

    bins = (0, .2, 3)
    group_names = ['Cant Afford', 'Can Afford']
    dataSetUp[DependentVariable] = pd.cut(dataSetUp[DependentVariable], bins, labels=group_names)
    dataSetUp.to_csv('test.csv')
    label_quality = LabelEncoder()
    dataSetUp[DependentVariable] = label_quality.fit_transform(dataSetUp[DependentVariable])
    showDataHeadAndInfo(head_Value)
    print(dataSetUp[DependentVariable].value_counts())
    sns.set_theme(style="darkgrid")
    sns.countplot(y=dataSetUp[DependentVariable])


# Load Data from CSV
loadAndExtractData()

# Print Info
showDataHeadAndInfo(head_Value)

# preProcessing
preProcessing()

'''
seperate dependent and independent variables
'''

X = dataSetUp.drop(DependentVariable, axis=1)
y = dataSetUp[DependentVariable]

# Train and test with random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomstate)

# Optimizing with standardScaler to minimize bias and normalize values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# print(X_train[:10])
dict_classifiers = {
    "rfc": RandomForestClassifier(n_estimators=200),
    "clf": svm.SVC(),
    "mlpc": MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=500,  random_state=1)
}

for model, model_instantiation in dict_classifiers.items():
    model = model_instantiation
    model.fit(X_train, y_train)
    y_score = model.predict(X_test)

    #todo
    # notes from Class
    # Summarize model.fit(X_train, y_train)
    # Find P values if P greater than .05 discard variable
    # Gains and Lift Chart
    # multicollinearity VIF calculator


    confusion_Matrix = confusion_matrix(y_test, y_score)
    cm = accuracy_score(y_test, y_score)
    print(f"Printing Model details for : {model}\n"
          f"Printing Confusion Matrix\n{confusion_Matrix}\n"
          f"Printing Classification Report\n {classification_report(y_test, y_score)}\n"
          f"****\n"
          f"End of Model\n"
          f"****\n")

    Xnew = [[7008.0, 22048, 3.21]]
    Xnew = sc.transform(Xnew)
    mlpc = MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=500,  random_state=1)
    mlpc.fit(X_train, y_train)
    pred_mlpc = mlpc.predict(X_test)
    ynew = mlpc.predict(Xnew)
    if ynew == 1:
        print(f"I am too poor to afford food")
    else:
        print(f"I will survive")


Xnew = [[4308.0, 0, 0]]
ynew = 0
cost = 18000
famSize = 3
reduceCostIteratorValue = 200
while ynew == 0:

   # XnewElement = Xnew[0:1][0][0] - 0
    Xnew = [[Xnew[0:1][0][0] - 0, cost, famSize]]
    Xnew1 = sc.transform(Xnew)
   # mlpc = MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=500)
    mlpc.fit(X_train, y_train)
    ynew = mlpc.predict(Xnew1)
    print(f'new ynew: {ynew} for xnew: {Xnew}')
    if ynew == 1:
        print(f"VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV You will Be fine at current cost below "
              f"VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
        break
    else:
        print(f"HMM you are spending too much iterating to find you an optimal amount trying {Xnew} ")
        Xnew = [[Xnew[0:1][0][0] - reduceCostIteratorValue, cost, famSize]]
        Xnew1 = sc.transform(Xnew)
        #mlpc = MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=500,  random_state=1)
        mlpc.fit(X_train, y_train)
        ynew = mlpc.predict(Xnew1)

print(f"***********************************************-- "
      f"We suggest you reduce your annual expenditure to ${Xnew[0][0]} for family counts of {Xnew[0][2]} and income of ${Xnew[0][1]}"
      f" --***********************************************")
plt.show()
