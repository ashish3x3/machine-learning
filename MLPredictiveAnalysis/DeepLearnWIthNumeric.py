import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
import os
from sklearn.cross_validation import train_test_split
style.use("ggplot")
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.decomposition import RandomizedPCA
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
# from scipy import sparse

listToSend = []

# path = "C:\Users\lenovo\Downloads\Compressed\LoanStats3a.csv_2\csv"
# pathToFiles = "C:\Users\lenovo\Downloads\Compressed\DataSet\CsvDataFiles"
# pathTo2CsvFiles = "C:\Users\lenovo\Downloads\Compressed\DataSet\CsvDataFiles\Twocsvfiles"
pathToSave = "InputFiles\custom_predictor_files"

FEATURES = ['loan_status',
            'loan_amnt',
            'term',
            'int_rate',
            'grade',
            'sub_grade',
            'emp_length',
            'home_ownership',
            'annual_inc',
            'verification_status',
            'purpose',
            'zip_code',
            'addr_state',
            'dti',
            'delinq_2yrs',
            'inq_last_6mths',
            'mths_since_last_delinq',
            'open_acc',
            'total_acc',
            'out_prncp',
            'acc_now_delinq'
            ]


data_df = pd.DataFrame.from_csv(os.path.join(pathToSave, "PredictionWithHeterogeneousDataSet.csv"), index_col=None)
data_df2 = pd.DataFrame.from_csv(os.path.join(pathToSave, "googlePredictionToUploadAfterDowngrading.csv"), index_col=None)

df_dummy = pd.get_dummies(data=data_df2, columns=['grade', 'sub_grade','home_ownership','verification_status','purpose',
                                       'addr_state'])
# data_df2['df_dummy'] = df_dummy
print df_dummy
print data_df2

data_df2 = data_df2[:10000]

# data_df = data_df.reindex(np.random.permutation(data_df.index))
data_df = data_df.replace("NaN", -1).replace("N/A", -1).replace("n/a",-1)

data_df_train, data_df_test = cross_validation.train_test_split(data_df, train_size=0.6)

dv = DictVectorizer(sparse=False)

df = pd.DataFrame(data_df).convert_objects(convert_numeric=True)
vec_x_cat_train = dv.fit_transform(df.to_dict(orient='records'))
print 'vec_x_cat_train ', vec_x_cat_train
print 'dv.feature_names_ ',dv.feature_names_

X_dum = pd.get_dummies(data_df)
print X_dum


def Kmeans():

    cluster = KMeans(n_clusters=8, init='k-means', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                     verbose=0, random_state=None, copy_x=True, n_jobs=1)

    cluster.fit(data_df_train)
    result = cluster.predict(data_df_test)

    print result


def KmeansWithPlot():

    kmeans_model = KMeans(n_clusters=5, random_state=1)
    good_columns = data_df._get_numeric_data()

    kmeans_model.fit(good_columns)
    labels = kmeans_model.labels_

    pca = PCA(n_components = 2)
    plot_columns = pca.fit_transform(good_columns)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
    plt.show()

    print 'corrr.....',data_df.corr()["loan_status"]

    candidate_columns = list(data_df.select_dtypes(include=[np.number]).columns.values)
    candidate_columns.remove("loan_status")
    target = "loan_status"

    train = data_df.sample(frac=0.8, random_state=1)
    test = data_df.loc[~data_df.index.isin(train.index)]
    print(train.shape)
    print(test.shape)

    model = LinearRegression()
    model.fit(train[candidate_columns], train[target])

    predictions = model.predict(test[candidate_columns])

    mean_sqr_error = mean_squared_error(predictions, test[target])
    print 'mean square error ',round(mean_sqr_error,2)


def KmeansWithPCA():
    print 'listToSend ',listToSend
    data_df = pd.DataFrame.from_csv(os.path.join(pathToSave, "PredictionWithHeterogeneousDataSet.csv"), index_col=None)
    data_df = data_df[pd.notnull(data_df['emp_length'])]
    data_df = data_df.dropna(how='all')
    data_df = data_df.replace("NaN", -1).replace("N/A", -1).replace("n/a", -1)
    X = np.array(data_df[FEATURES].values)
    print X.shape
    y = np.array(data_df["loan_status"].values)    #.reshape(66730, 1)
    print y.shape
    print y
    data_combined = np.column_stack((X, y))
    print(data_combined)

    data_norm = normalize(data_combined)
    print(data_norm)

    pca = RandomizedPCA(n_components=20, whiten=True)
    pca.fit(data_norm)
    X = pca.transform(data_norm)

    # Normalize subset of data
    data_1_norm = normalize(X)
    print(data_1_norm)

    print 'PCA transform ..'
    # pca.transform(data_1_norm)

    kmeans_model = KMeans(n_clusters=5, random_state=1)
    good_columns = data_df._get_numeric_data()

    kmeans_model.fit(good_columns)
    labels = kmeans_model.labels_
    print 'labels ',labels

    print 'correlation.....', data_df.corr()["loan_status"]
    candidate_columns = list(data_df.select_dtypes(include=[np.number]).columns.values)
    candidate_columns.remove("loan_status")
    target = "loan_status"
    train = data_df.sample(frac=0.8, random_state=1)
    test = data_df.loc[~data_df.index.isin(train.index)]
    print(train.shape)
    print(test.shape)

    #########################################

    model = LinearRegression()
    model.fit(train[candidate_columns], train[target])
    print 'test[candidate_columns] ',test[candidate_columns]
    predictions = model.predict(test[candidate_columns])
    # X = pd.DataFrame(listToSend)
    # predictions = model.predict(X)

    # print 'model prediction ',predictions
    # print 'Test.terget ',test[target]

    mean_sqr_error = mean_squared_error(predictions, test[target])
    print 'mean square error ', round(mean_sqr_error, 2)

    #########################################

    model2 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
    # Fit the model to the data.
    model2.fit(train[candidate_columns], train[target])
    # Make predictions.
    predictions2 = model2.predict(test[candidate_columns])
    # predictions2 = model2.predict(listToSend)
    # Compute the error.
    mean_sqr_error2 = mean_squared_error(predictions2, test[target])

    # print 'mean squr error 2 ', mean_sqr_error2
    print 'mean squre error 2',round(mean_sqr_error2,2)

    #########################################
    num_folds = 10
    num_instances = len(X)
    seed = 7
    num_trees = 100
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
    # X = np.array(listToSend)
    results = cross_validation.cross_val_score(model, X, y, cv=kfold)
    print 'result.mean() ',results.mean()


    ##########################################

    num_folds = 10
    num_instances = len(X)
    seed = 7
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    # create the sub models
    estimators = []
    model1 = LogisticRegression()
    estimators.append(('logistic', model1))
    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))
    model3 = SVC()
    estimators.append(('svm', model2))
    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    # X = np.array(listToSend)
    results = cross_validation.cross_val_score(ensemble, X, y, cv=kfold)
    print(results.mean())


    plt.figure('Reference Plot')
    plt.scatter(x=X[:, 0], y=X[:, 1], c=labels)
    plt.legend(labels)
    plt.title('Loan Application Distribution')
    plt.show()



def Build_Data_Set():
    data_df = pd.DataFrame.from_csv(os.path.join(pathToSave, "PredictionWithHeterogeneousDataSet.csv"),index_col=None)

    # data_df = data_df[:1000]
    data_df = data_df.reindex(np.random.permutation(data_df.index))
    data_df = data_df.replace("NaN", -1).replace("N/A", -1).replace("n/a",-1)

    X = np.array(data_df[FEATURES].values)  # .tolist())
    print 'X ',X

    y = np.array(data_df["loan_status"] )
    print 'y ', y

    # candidate_columns = list(data_df.select_dtypes(include=[np.number]).columns.values)
    # print candidate_columns

    X = preprocessing.scale(X)
    print 'X scale ', X

    return X, y


def Analysis():

    test_size = 1000
    X, y = Build_Data_Set()
    print(len(X))
    print(len(y))

    clf = svm.SVC(kernel="linear", C=1.0)
    # clf.fit(X[:-test_size],y[:-test_size])

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33)

    print '-',train_x, test_x, train_y, test_y

    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)

    correct_count = 0

    my_metrics = metrics.classification_report(test_y, predictions)
    print my_metrics
    print metrics.confusion_matrix(test_y, predictions)

    # for x in range(1, test_size  1):
    #     print 'x ',x
    #     print 'X[-x] ',X[-x]
    #     print 'y[-x] ', y[-x]
    #     if clf.predict(X[-x])[0] == y[-x]:
    #         correct_count = 1
    #
    # print("Accuracy:", (correct_count / test_size) * 100.00)


# Analysis()
# KmeansWithPlot()
# Kmeans()
listToSend = [2323.0, 2.0, 2.0, u'A', u'A3', 23.0, u'A', u'MORTAGE',
              2323.0, u'Verified', u'vacation', u'23233', u'KS', 23.67, 23.0,
              'December-16', 23.0, 23.0, 2.0, 2323.0, 232.0, 2323.0
            ]
KmeansWithPCA()