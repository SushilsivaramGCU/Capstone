#!/usr/bin/env python

__author__ = "Sushil Sivaram, Megha Gubbala"
__copyright__ = "N/A"
__credits__ = ["Isac Artzi", "Dinesh Sthapit", "Ken Ferrell", "James Dzikunu", "Tracy Roth", "Renee Morales"]
__license__ = "ECL"
__maintainer__ = "Sushil Sivaram, Megha Gubbala"
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
# CSVData = os.getenv('CSVData')
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
    keepcolumns = ['cost_yr', 'median_income', 'affordability_ratio', 'ave_fam_size']
    dataSetUp = dataSetUp.filter(keepcolumns)
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
    print(dataSetUp['ave_fam_size'])

def plotting():
    histmedian_income = dataSetUp['median_income'].plot.hist(bins=50, alpha=0.5)
    histmedian_income.figure.savefig('histMedianIncome.png')
    hist_avg_fam = dataSetUp['ave_fam_size'].plot.hist(bins=5, alpha=0.05)
    hist_avg_fam.figure.savefig('histavgFamsize.png')
    hist_Cost_yr = dataSetUp['cost_yr'].plot.hist(bins=50, alpha=0.05)
    hist_Cost_yr.figure.savefig('histCost.png')



    #hist_Cost_yr.savfig('cost_yr.pdf')



# Load Data from CSV
loadAndExtractData()

# Print Info
showDataHeadAndInfo(head_Value)

# Graph
plotting()
