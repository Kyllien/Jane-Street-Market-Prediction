import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/33640/OneDrive/Bureau/train.csv')
feat = pd.read_csv('C:/Users/33640/OneDrive/Bureau/features.csv')

#Function for create a dictionnary with 1 dataframe = 1 day
def dict_by_day (df) :
    d = {}
    for i in range (max(df['date']) + 1):
        c = 'date ==' + str(i)
        d[i] = df.query(c)
    return(d)


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