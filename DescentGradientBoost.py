import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as skms
from sklearn import ensemble as ske
import GestionData as gd


def GradientBoost(train,action):

    #Selection d'une base plus petite car trop grande pour effectuer les calculs au vu de la puissance de notre ordinateur
    #On selectionne sur les 50 premiers jours pour avoir mini 100k d'observations, ce qui correspond à 1/10 de la base


    df2 = df[df['date']<=49]

    train, action = pr_action(df2)

    #Random Cross Validation
    X_train, X_test, y_train, y_test = skms.train_test_split(train, action, test_size=0.3, random_state=0)

    #GradientBoostingClassifier

    import time
    start_time = time.time()
    clf = ske.GradientBoostingClassifier(n_estimators=2000, learning_rate=1.0, max_depth=7, random_state=0).fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Accuracy score (training): {0:.3f}".format(clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(clf.score(X_test, y_test)))
    #Resultat pour : n_estimators=500, learning_rate=1.0, max_depth=1, random_state=0
    #Accuracy score (training): 0.569
    #Accuracy score (validation): 0.554

    #Resultat pour : n_estimators=1000, learning_rate=1.0, max_depth=1, random_state=0
    #Accuracy score (training): 0.577
    #Accuracy score (validation): 0.556

    #Resultat pour : n_estimators=500, learning_rate=1.0, max_depth=3, random_state=0
    #Accuracy score (training): 0.70
    #Accuracy score (validation): 0.62

    #Resultat pour : n_estimators=500, learning_rate=3.0, max_depth=1, random_state=0
    #Accuracy score (training): 0.471
    #Accuracy score (validation): 0.475

    #Resultat pour : n_estimators=500, learning_rate=0.8, max_depth=1, random_state=0
    #Accuracy score (training): 0.566
    #Accuracy score (validation): 0.553

    #Resultat pour : n_estimators=500, learning_rate=0.5, max_depth=1, random_state=0
    #Accuracy score (training): 0.561
    #Accuracy score (validation): 0.551

    #Resultat pour : n_estimators=500, learning_rate=1, max_depth=7, random_state=0
    #Accuracy score (training): 0.985
    #Accuracy score (validation): 0.663

    #Resultat pour : n_estimators=500, learning_rate=1, max_depth=5, random_state=0
    #Accuracy score (training): 0.855
    #Accuracy score (validation): 0.647

    return clf

