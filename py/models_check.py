#gives model report in dataframe
def model_report(model,training_x,testing_x,training_y,testing_y,name) :
    model.fit(training_x,training_y)
    predictions  = model.predict(testing_x)
    accuracy     = accuracy_score(testing_y,predictions)
    recallscore  = recall_score(testing_y,predictions)
    precision    = precision_score(testing_y,predictions)
    roc_auc      = roc_auc_score(testing_y,predictions)
    f1score      = f1_score(testing_y,predictions) 

    
    df = pd.DataFrame({"Model"           : [name],
                       "Accuracy_score"  : [accuracy],
                       "Recall_score"    : [recallscore],
                       "Precision"       : [precision],
                       "f1_score"        : [f1score],
                       "AUC"             : [roc_auc],
                      })
    return df

def split_data(df):
    # Splitting the data into test and train sets
    X = df.drop(['churn'], axis=1)
    y = df['churn']
    # make a test-train split
    split_size = 0.4
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split_size,random_state=63) 
    
    # check split data diemention  
    #print("Train dataset: {0}{1}".format(X_train.shape, y_train.shape))
    #print("Test dataset: {0}{1}".format(X_test.shape, y_test.shape))
    
    #scale features 
    ss = StandardScaler()
    # Scale the train and test data
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    return X_train, X_test, y_train, y_test


def model_report_fit(df, s = False):
# Let's create a baseline model for all models. Let's statr with as minimum amount of parameters as possible 
    if s:
        smote = SMOTE()
        X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)
    else: X_train, X_test, y_train, y_test = split_data(df)
# LogisticRegression
    name = "Logistic Regression"
    logit = LogisticRegression()
    #X_train, X_test, y_train, y_test = split_data(df)
    model1 = model_report(logit,X_train, X_test, y_train, y_test,name)

# DecisionTree Classifier
    name = "Decision Tree"
    decision_tree = DecisionTreeClassifier(random_state = 0)
    #X_train, X_test, y_train, y_test = split_data(df)
    model2 = model_report(decision_tree,X_train, X_test, y_train, y_test,name)

# KNeighborsClassifier
    name = "KNN Classifier"
    knn = KNeighborsClassifier(n_neighbors=3)    
    #X_train, X_test, y_train, y_test = split_data(df)
    model3 = model_report(knn,X_train, X_test, y_train, y_test,name)
# Naive Bayes
    #name = "Naive Bayes"
    #gnb = GaussianNB(priors=None)
    #X_train, X_test, y_train, y_test = split_data(df)
    #model4 = model_report(gnb,X_train, X_test, y_train, y_test,name)

    name = "Random Forest Classifier"
    rfc = RandomForestClassifier(random_state=0)    
    #X_train, X_test, y_train, y_test = split_data(df)
    model5 = model_report(rfc,X_train, X_test, y_train, y_test,name)

    name = "SVM Classifier Linear"
    svc  = SVC(gamma='auto', kernel='linear')
    #X_train, X_test, y_train, y_test = split_data(df)
    model6 = model_report(svc,X_train, X_test, y_train, y_test,name)

    name = "AdaBoost Classifier"
    ada = AdaBoostClassifier()
    #X_train, X_test, y_train, y_test = split_data(df)
    model7 = model_report(ada,X_train, X_test, y_train, y_test,name)

#concat all models
    model_performances = pd.concat([model1,model2,model3,#model4,
                                model5,model6,model7],axis = 0).reset_index()

    model_performances = model_performances.drop(columns = "index",axis =1)
    return model_performances