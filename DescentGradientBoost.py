


def GradientBoost(train,action):

    #Selection d'une base plus petite car trop grande pour effectuer les calculs au vu de la puissance de notre ordinateur
    #On selectionne sur les 50 premiers jours pour avoir mini 100k d'observations, ce qui correspond à 1/10 de la base totale

    df2 = df[df['date']<=49]

    train, action = pr_action(df2)

    

    ##Verif par parametre
    #En fonction de learning rate
    scores = experiment([
    ske.GradientBoostingClassifier(max_depth=1, n_estimators=10, learning_rate=0.6),
    ske.GradientBoostingClassifier(max_depth=1, n_estimators=10, learning_rate=0.8),
    ske.GradientBoostingClassifier(max_depth=1, n_estimators=10, learning_rate=1),
    ],train,action)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot([_[0] for _ in scores], label="GBC(1,lr=0.6)")
    ax.plot([_[1] for _ in scores], label="GBR(1, lr=0.8)")
    ax.plot([_[2] for _ in scores], label="GBR(1, lr=1)")
    ax.set_title("Comparaison pour différents learning_rate et des fonctions en escalier")
    ax.legend();
    plt.show()

    #En fonction de n_estimators
    scores = experiment([
    ske.GradientBoostingClassifier(max_depth=1, n_estimators=50, learning_rate=1),
    ske.GradientBoostingClassifier(max_depth=1, n_estimators=100, learning_rate=1),
    ske.GradientBoostingClassifier(max_depth=1, n_estimators=200, learning_rate=1),
    ske.GradientBoostingClassifier(max_depth=1, n_estimators=300, learning_rate=1),
    ],train,action)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot([_[0] for _ in scores], label="GBC(1, n_est1=50)")
    ax.plot([_[1] for _ in scores], label="GBR(1, n_esti=100)")
    ax.plot([_[2] for _ in scores], label="GBR(1, n_esti=200)")
    ax.plot([_[3] for _ in scores], label="GBR(1, n_esti=300)")
    ax.set_title("Comparaison pour différents nombres d'estimateurs et des fonctions en escalier")
    ax.legend();
    plt.show()

    #En fonction de la profondeur des arbres
    scores = experiment([
    ske.GradientBoostingClassifier(max_depth=1, n_estimators=100, learning_rate=1),
    ske.GradientBoostingClassifier(max_depth=2, n_estimators=100, learning_rate=1),
    ske.GradientBoostingClassifier(max_depth=4, n_estimators=100, learning_rate=1),
    ske.GradientBoostingClassifier(max_depth=8, n_estimators=100, learning_rate=1),
    ],train,action)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot([_[0] for _ in scores], label="GBC(1, max_depth=1)")
    ax.plot([_[1] for _ in scores], label="GBR(1, max_depth=2)")
    ax.plot([_[2] for _ in scores], label="GBR(1, max_depth=4)")
    ax.plot([_[3] for _ in scores], label="GBR(1, max_depth=8)")
    ax.set_title("Comparaison pour différentes profondeurs d'arbres et des fonctions en escalier")
    ax.legend();
    plt.show()


    #Random Cross Validation
    X_train, X_test, y_train, y_test = skms.train_test_split(train, action, test_size=0.3, random_state=0)

    #GradientBoostingClassifier

    import time
    start_time = time.time()
    clf = ske.GradientBoostingClassifier(n_estimators=2000, learning_rate=0.01, max_depth=7, random_state=0).fit(X_train, y_train)
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

