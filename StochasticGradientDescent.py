



#Utlise la fonction action qui permet de creer le Y
train, action = pr_action(df)

#Random Cross Validation
X_train, X_test, y_train, y_test = skms.train_test_split(train, action, test_size=0.3, random_state=0)

##Verif par parametre
#En fonction de max_iter
scores = experiment([
SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
SGDClassifier(loss="hinge", penalty="l2", max_iter=10),
SGDClassifier(loss="hinge", penalty="l2", max_iter=15),
SGDClassifier(loss="hinge", penalty="l2", max_iter=20),
SGDClassifier(loss="hinge", penalty="l2", max_iter=30),
SGDClassifier(loss="hinge", penalty="l2", max_iter=50),
SGDClassifier(loss="hinge", penalty="l2", max_iter=80),
],train,action)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot([_[0] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=5)")
ax.plot([_[1] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=10)")
ax.plot([_[2] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=15)")
ax.plot([_[3] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=20)")
ax.plot([_[4] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=30)")
ax.plot([_[5] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=50)")
ax.plot([_[6] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=80)")
ax.set_title("Comparaison pour différents maximum d'itérations et des fonctions en escalier")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
plt.show()

#En fonction de de la fonction de cout
scores = experiment([
SGDClassifier(loss="hinge", penalty="l2", max_iter=30),
SGDClassifier(loss="loss", penalty="l2", max_iter=30),
SGDClassifier(loss="modified_huber", penalty="l2", max_iter=30),
SGDClassifier(loss="hinge", penalty="l2", max_iter=50),
SGDClassifier(loss="loss", penalty="l2", max_iter=50),
SGDClassifier(loss="modified_huber", penalty="l2", max_iter=50),
],train,action)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot([_[0] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=30)")
ax.plot([_[1] for _ in scores], label="SGD(loss='loss', penalty='l2',max_iter=30)")
ax.plot([_[2] for _ in scores], label="SGD(loss='modified_huber', penalty='l2',max_iter=30)")
ax.plot([_[3] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=50)")
ax.plot([_[4] for _ in scores], label="SGD(loss='loss', penalty='l2',max_iter=50)")
ax.plot([_[5] for _ in scores], label="SGD(loss='modified_huber', penalty='l2',max_iter=50)")
ax.set_title("Comparaison pour différents maximum d'itérations et des fonctions en escalier")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
plt.show()

#En fonction de de la fonction de cout
scores = experiment([
SGDClassifier(loss="hinge", penalty="l2", max_iter=30),
SGDClassifier(loss="log", penalty="l2", max_iter=30),
SGDClassifier(loss="modified_huber", penalty="l2", max_iter=30),
SGDClassifier(loss="hinge", penalty="l2", max_iter=50),
SGDClassifier(loss="log", penalty="l2", max_iter=50),
SGDClassifier(loss="modified_huber", penalty="l2", max_iter=50),
],train,action)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot([_[0] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=30)")
ax.plot([_[1] for _ in scores], label="SGD(loss='log', penalty='l2',max_iter=30)")
ax.plot([_[2] for _ in scores], label="SGD(loss='modified_huber', penalty='l2',max_iter=30)")
ax.plot([_[3] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=50)")
ax.plot([_[4] for _ in scores], label="SGD(loss='log', penalty='l2',max_iter=50)")
ax.plot([_[5] for _ in scores], label="SGD(loss='modified_huber', penalty='l2',max_iter=50)")
ax.set_title("Comparaison pour différentes fonctions de couts et des fonctions en escalier")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
plt.show()


#En fonction de de la fonction de penalty
scores = experiment([
SGDClassifier(loss="hinge", penalty="l2", max_iter=50),
SGDClassifier(loss="hinge", penalty="l1", max_iter=50),
SGDClassifier(loss="hinge", penalty="elasticnet", max_iter=50),
SGDClassifier(loss="modified_huber", penalty="l2", max_iter=50),
SGDClassifier(loss="modified_huber", penalty="l1", max_iter=50),
SGDClassifier(loss="modified_huber", penalty="elasticnet", max_iter=50),
],train,action)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot([_[0] for _ in scores], label="SGD(loss='hinge', penalty='l2',max_iter=50)")
ax.plot([_[1] for _ in scores], label="SGD(loss='hinge', penalty='l1',max_iter=50)")
ax.plot([_[2] for _ in scores], label="SGD(loss='hinge', penalty='elasticnet',max_iter=50)")
ax.plot([_[3] for _ in scores], label="SGD(loss='modified_huber', penalty='el2',max_iter=50)")
ax.plot([_[4] for _ in scores], label="SGD(loss='modified_huber', penalty='l1',max_iter=50)")
ax.plot([_[5] for _ in scores], label="SGD(loss='modified_huber', penalty='elasticnet',max_iter=50)")
ax.set_title("Comparaison pour différents penalty et des fonctions en escalier")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
plt.show()

clf = SGDClassifier(loss="hinge", penalty="l1", max_iter=50)
clf.fit(X_train, y_train)
print("Accuracy score (training): {0:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(clf.score(X_test, y_test)))

dump(clf, 'SGD.joblib')
