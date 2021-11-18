import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stl
from sklearn.linear_model import SGDClassifier
from joblib import dump, load
from numpy import nan
from sklearn import model_selection as skms
from sklearn import ensemble as ske
from tqdm import tqdm


#Base de donnée

df = pd.read_csv('C:/Users/33640/OneDrive/Bureau/train.csv')
#feat = pd.read_csv('C:/Users/33640/OneDrive/Bureau/features.csv')

 
#Function for describe all the column and put this in a dictionary
def describe_dico(df) :
    d = {}
    for i in range (len(df.columns)) :
        d[df.columns[i]] = df.iloc[:,[i]].describe()
    return d

#Regarde stationnarite et si loi normale pour les variables, du moins type de distribution
# 

#ACF for feature PROBLEM
def acf (df) :
    d = {}
    for i in range (len(df.columns)) :
        d[df.columns[i]] = pd.DataFrame(stl.acf(df.iloc[:,[i]].dropna()))
    return d

#PACF to do
def pacf (df) :
    d = {}
    for i in range (len(df.columns)) :
        d[df.columns[i]] = pd.DataFrame(stl.pacf(df.iloc[:,[i]].dropna()))
    return d

#Stationarite with kpss : Kwiatkowski-Phillips-Schmidt-Shin test
#Null Hypothesis : The data is stationary around a constant
def kpss (df) :
    d={}
    for i in range(len(df.columns)) :
        d[df.columns[i]] = stl.kpss(df.iloc[:,[i]].dropna())[1]
    return d



def main () :
    print(df.head())
    
    print(df.info())
    
    print(df.dtypes)
    
    # colonne = df.columns
    # date = df['date']
    # ts = df['ts_id']
    # h = df.head()
    # date.describe()
    # plt.scatter(df["ts_id"],df["feature_1"])
    # feature1 = df["feature_1"]
    # feature1.describe()
    # test = df.loc[:,['feature_1','ts_id']]
    
    d = dict_by_day(df)
    m = d[0]
    decris = describe_dico(df)
    

    #ACF/PACF
    d_acf = acf(df.iloc[:,7:137])
    d_pacf = pacf(df.iloc[:,7:13])


    #Stationarite
    stationnarite_k = kpss(df.iloc[:,7:137]) #Long
    # stationnarite_a = stl.adfuller(df.iloc[:,7:8]) #Df too big for this test






