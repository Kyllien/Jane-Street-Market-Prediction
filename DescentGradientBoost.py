import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as skms
from sklearn import ensemble as ske
import GestionData as gd


def GradientBoost(train,action):
    #Random Cross Validation
    X_train, X_test, y_train, y_test = skms.train_test_split(train, action, test_size=0.3, random_state=0)

    #GradientBoostingClassifier
    import time
    start_time = time.time()
    clf = ske.GradientBoostingClassifier(n_estimators=500, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return clf

