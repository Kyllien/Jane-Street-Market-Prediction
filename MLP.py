from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import model_selection as skms


def MLP(df) :

    df2 = df[df['date']<=49]

    train, action = pr_action(df2)

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(action)
    encoded_Y = encoder.transform(action)


    # baseline model
    def create_baseline():
	    # create model
	    model=Sequential()
	    model.add(Dense(130, input_dim=130, activation='relu'))
	    model.add(Dense(1, activation='sigmoid'))
	    # Compile model
	    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	    return model
    # evaluate model with standardized dataset
    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, train, encoded_Y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    #Avec build_fn=create_baseline, epochs=100, batch_size=5, verbose=0 et n_splits = 10 
    #Resultat 57.4%, sd = 0.24%


    return model