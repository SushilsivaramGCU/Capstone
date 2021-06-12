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
    plt.figure()
    histmedian_income = dataSetUp['median_income'].plot.hist(bins=25, grid=True, rwidth=0.9, color='#607c8e')
    plt.title('Histogram of Median Income')
    plt.xlabel('Median Income in $')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.5)
    histmedian_income.figure.savefig('.\outputs\histMedianIncome.png')

    plt.figure()
    hist_avg_fam = dataSetUp['ave_fam_size'].plot.hist(bins=25,  grid=True, rwidth=0.9, color='#607c8e')
    plt.title('Histogram of Family Size')
    plt.xlabel('Family Size')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.5)
    hist_avg_fam.figure.savefig('.\outputs\histavgFamsize.png')

    plt.figure()

    hist_Cost_yr = dataSetUp['cost_yr'].plot.hist(bins=25, grid=True, rwidth=0.9, color='#607c8e')
    plt.title('Histogram of Yearly Cost of Food $')
    plt.xlabel('Yearly Cost $')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.5)
    plt.figure(figsize=(8, 6))
    hist_Cost_yr.figure.savefig('.\outputs\histCost.png')

    plt.figure()
    scattermedian_income = dataSetUp.plot.scatter(c='DarkBlue', x='median_income', y = 'cost_yr' )
    plt.title('scatterogram of Median Income vs Expenditure')
    plt.xlabel('Median Income in $')
    plt.ylabel('cost_yr')
    plt.grid(axis='y', alpha=0.5)
    scattermedian_income.figure.savefig('.\outputs\scatterMedianIncomeVSExpenditure.png')

    plt.figure()
    scattermedian_income = dataSetUp.plot.scatter(c='DarkBlue', x='ave_fam_size', y = 'cost_yr' )
    plt.title('scatterogram of Family Size vs Expenditure')
    plt.xlabel('Family Size')
    plt.ylabel('cost_yr')
    plt.grid(axis='y', alpha=0.5)
    scattermedian_income.figure.savefig('.\outputs\scatterFamSizeVSExpenditure.png')

# Load Data from CSV
loadAndExtractData()

# Print Info
showDataHeadAndInfo(head_Value)

# Graph
plotting()
