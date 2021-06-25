#!/usr/bin/env python

__author__ = "Sushil Sivaram, Megha Gubbala", "Sylvia Nanyangwe"
__copyright__ = "N/A"
__credits__ = ["Isac Artzi", "Dinesh Sthapit", "Ken Ferrell", "James Dzikunu", "Tracy Roth", "Renee Morales"]
__license__ = "ECL"
__maintainer__ = "Sushil Sivaram, Megha Gubbala", "Sylvia Nanyangwe"
__email__ = "SushilSivaram@gmail.com"
__status__ = "Development"

from matplotlib import pyplot
from scipy import stats
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from yellowbrick.features import Rank2D
from yellowbrick.target import ClassBalance
from yellowbrick.classifier import ConfusionMatrix
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import scikitplot as skplt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Define Variables
load_dotenv()
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
    global datasetupUnprocessed

    def readCSV():
        global dataSetUp, datasetupUnprocessed
        dataSetUp = pd.read_csv(CSVData)
        datasetupUnprocessed = dataSetUp

    readCSV()
    keepcolumns = [#'LL95_affordability_ratio', 'UL95_affordability_ratio',
                   'race_eth_code','median_income','affordability_ratio', 'ave_fam_size']
    dataSetUp =dataSetUp.filter(keepcolumns)
    for keep in keepcolumns:
        dataSetUp = dataSetUp[dataSetUp[keep].notna()]
    dataSetUp= dataSetUp[(np.abs(stats.zscore(dataSetUp)) < 3).all(axis=1)]
    dataSetUp.to_csv('cleaned.csv', index=False)
    print (dataSetUp.shape)





#print Info
def showDataHeadAndInfo(data,headCount):
    print(f"showing head {headCount} values")
    print(data.head(headCount))
    print("**********")
    print("Showing info of dataset")
    print(data.describe(include='all'))


# preProcessing
def preProcessing():
    bins = (0, .2, 5)
    group_names = ['Cant Afford', 'Can Afford']
    dataSetUp[DependentVariable] = pd.cut(dataSetUp[DependentVariable], bins, labels=group_names)
    dataSetUp.to_csv('test.csv')
    label_quality = LabelEncoder()
    dataSetUp[DependentVariable] = label_quality.fit_transform(dataSetUp[DependentVariable])
    #showDataHeadAndInfo(head_Value)
    print(dataSetUp[DependentVariable].value_counts())


#plotting
def plotting(dataSetUp, state):
    plt.figure()
    histmedian_income = dataSetUp['median_income'].plot.hist(bins=25, grid=True, rwidth=0.9, color='#607c8e')
    plt.title(f'Histogram of Median Income {state}')
    plt.xlabel('Median Income in $')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.5)
    histmedian_income.figure.savefig(f'.\outputs\histMedianIncome{state}.png')

    plt.figure()
    hist_avg_fam = dataSetUp['ave_fam_size'].plot.hist(bins=25,  grid=True, rwidth=0.9, color='#607c8e')
    plt.title(f'Histogram of Family Size {state}')
    plt.xlabel('Family Size')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.5)
    hist_avg_fam.figure.savefig(f'.\outputs\histavgFamsize{state}.png')

    plt.figure()
    hist_race_eth_name = dataSetUp['race_eth_code'].plot.hist(bins=2, grid=True, rwidth=0.9, color='#607c8e')
    plt.title(f'Histogram race distribution {state}')
    plt.xlabel('Race')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.5)
    hist_race_eth_name.figure.savefig(f'.\outputs\histCost{state}.png')

    plt.figure()
    scattermedian_income = dataSetUp.plot.scatter(c='DarkBlue', x='median_income', y = 'ave_fam_size' )
    plt.title(f'scatterogram of Median Income vs Family size {state}')
    plt.xlabel('Median Income in $')
    plt.ylabel('ave_fam_size')
    plt.grid(axis='y', alpha=0.5)
    scattermedian_income.figure.savefig(f'.\outputs\scatterMedianIncomeVSFamilySize{state}.png')

    plt.figure()

def trainDataset():
    global X_train
    global y_train
    global X_test
    global y_test
    global sc
    # Train and test with random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomstate)
    # Optimizing with standardScaler to minimize bias and normalize values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

def predictorImportance():
    X = dataSetUp.drop(DependentVariable, axis=1)
    y = dataSetUp[DependentVariable]
    model = LogisticRegression()
    # fit the model
    model.fit(X, y)
    importance = model.coef_[0]
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


def vifCheck(dataSetUp):
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = dataSetUp.columns
    vif_data["VIF"] = [variance_inflation_factor(dataSetUp.values, i)
                       for i in range(len(dataSetUp.columns))]
    print(vif_data)



# Load Data from CSV
loadAndExtractData()

print(datasetupUnprocessed.corr())



# Print Info
showDataHeadAndInfo(datasetupUnprocessed,head_Value)

# Exploratory plotting
plotting(datasetupUnprocessed , "BeforeProcessing")

# preProcessing
preProcessing()





showDataHeadAndInfo(dataSetUp,head_Value)
plotting(dataSetUp, "PostProcessing")
vifCheck(dataSetUp)

predictorImportance()

'''
seperate dependent and independent variables
'''

X = dataSetUp.drop(DependentVariable, axis=1)
y = dataSetUp[DependentVariable]
#yellow brick
visualizer = Rank2D(algorithm='pearson')
visualizer.fit(X, y)
visualizer.transform(X)
visualizer.show()
visualizer = ClassBalance(labels=["Cant Afford", "Can Afford"])
visualizer.fit(y)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

trainDataset()

from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
# print(X_train[:10])
dict_classifiers = {
    "rfc": RandomForestClassifier(n_estimators=200),
    "clf": svm.SVC(),
    "mlpc": MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=500,  random_state=1),
    "lr": LogisticRegression(),
}

for model, model_instantiation in dict_classifiers.items():
    model = model_instantiation
    model.fit(X_train, y_train)
    y_score = model.predict(X_test)
    # yellow brick
    cm = ConfusionMatrix(model, classes=[0,1])
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.show()
    confusion_Matrix = confusion_matrix(y_test, y_score)
    cm = accuracy_score(y_test, y_score)
    print(f"Printing Model details for : {model}\n"
          f"Printing Confusion Matrix\n{confusion_Matrix}\n"
          f"Printing Classification Report\n {classification_report(y_test, y_score)}\n"
          f"****\n"
          f"End of Model\n"
          f"****\n")

mlpc = MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=500,  random_state=randomstate)
mlpc.fit(X_train, y_train)

Xnew = X_test[[0]]
pred_mlpc = mlpc.predict(X_test)
ynew = mlpc.predict(Xnew)
if ynew == 0:
    print(f"I am too poor to afford food")
else:
    print(f"I will survive")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
nb = GaussianNB()
nb.fit(X_train, y_train)
predicted_probas = nb.predict_proba(X_test)
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()


Xnew = [[8000, 0, 0,]]
ynew = 0
income = 30000
famSize = 3
reduceCostIteratorValue = 50
while ynew == 0:
    Xnew = [[Xnew[0:1][0][0] - 0, income, famSize]]
    Xnew1 = sc.transform(Xnew)
    ynew = mlpc.predict(Xnew1)
    print(f'new ynew: {ynew} for xnew: {Xnew}')
    if ynew == 1:
        print(f"VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV You will Be fine at current cost below "
              f"VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
        break
    else:
        print(f"HMM you are spending too much iterating to find you an optimal amount trying {Xnew} ")
        Xnew = [[Xnew[0:1][0][0] - reduceCostIteratorValue, income, famSize]]
        Xnew1 = sc.transform(Xnew)
        ynew = mlpc.predict(Xnew1)

print(f"***********************************************-- "
      f"We suggest you reduce your annual expenditure to ${Xnew[0][0]} for family counts of {Xnew[0][2]} and income of ${Xnew[0][1]}"
      f" --***********************************************")

