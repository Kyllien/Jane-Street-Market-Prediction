

#Function for create a dictionnary with 1 dataframe = 1 day
def dict_by_day (df) :
    d = {}
    for i in range (max(df['date'])+1): 
        d[df.columns[i]] = df.query('date ==' + str(i))
    return(d)

d= df.query('date >=' + str(0) + ' && date < ' + str(10))

#Function for create a dictionnary with max(df['date'])/n dataframe = n day
def dict_by_nday (df,n) :
    d = {}
    n=10
    for i in range (int(max(df['date'])/n)+1):
        if(i*n> max(df['date'])):
            d[df.columns[i]] = df.query('date >=' + str(i*n))
        else:
            d[df.columns[i]] = df.query('date >=' + str(i*n) + "&& date < " + str((i+1)*n))
    return(d)

s=dict_by_nday(df,10)

def pr_action (df) :

    #Initilization of action (Take the trade or Not)
    #It's our Y
    action = ((df['resp']>0).astype('int'))

    #Initilisation of train data without date et ts_id
    train = df.iloc[:,7:137]

    #Put 0 for the NaN
    #Explication will be on the report
    train = train.replace(nan,0)
    action = action.replace(nan,0)

    return train, action






