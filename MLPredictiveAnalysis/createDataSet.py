import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier

path = "C:\Users\lenovo\Downloads\Compressed\LoanStats3a.csv_2\csv"
pathToFiles = "C:\Users\lenovo\Downloads\Compressed\DataSet\CsvDataFiles"
pathTo2CsvFiles = "C:\Users\lenovo\Downloads\Compressed\DataSet\CsvDataFiles\Twocsvfiles"
pathToSave = "A:\CSV"

def f(s):
    """ convert string to float if possible """
    if isinstance(s, str):
        # print 's is',s,len(s)
        s = s.strip()   # remove spaces at beginning and end of string
        # print 's after strip is',s, len(s)
        # print 'months in s ','months' in s

        if s.endswith('%'):  # remove %, if exists
            # print 's ends with %',s,len(s)
            s = s[:-1]
        elif 'months' in s:
            s= s.replace('months','').replace('+','').replace('>','')
            # print 'new s after removing months ',s
        elif 'years' in s:
            # print 'years in s ', 'years' in s
            s= s.replace('years','').replace('year','').replace('+','').replace('>','')
            # print 'new s after removing years ',s
        elif 'year' in s:
            # print 'year in s ', 'years' in s
            s= s.replace('year','').replace('+','').replace('>','').replace('<','')
            # print 'new s after removing years ',s

        try:
            # print 'return float of s ',s
            return float(s)
        except ValueError: # converting did not work
            # print 'exception normal s ',s
            return s  # return original string
    else:
        return s




fields = ['loan_status','loan_amnt','term', 'int_rate', 'grade', 'sub_grade', 'emp_length','home_ownership',
           'annual_inc','verification_status','purpose','zip_code','addr_state','dti','delinq_2yrs',
           'earliest_cr_line','inq_last_6mths','mths_since_last_delinq','open_acc','total_acc',
           'out_prncp','acc_now_delinq'
        ]

games = pd.read_csv(os.path.join(pathTo2CsvFiles, "LoanStats_2016Q1.csv"),nrows=500,skiprows=1,low_memory=False,usecols=fields)
#nrows=100
# print (games)
valid_loan_status = ['Charged Off','Default','Fully Paid','Current']
# print ('before shape',games.shape)
# print games["loan_status"]
games = games[games.loan_status.isin(valid_loan_status)]
# print games.loan_status
# print ('after shape',games.shape)

games = games.replace("NaN",-99999).replace("N/A",-99999).replace('n/a',-99999)
# print ('after drop na',games.shape)

# print (games.shape)
# print (games["int_rate"])
games = games.applymap(f)
# print games["loan_amnt"]
# print type(games.int_rate)
# intt_rate = games["int_rate"].map(f)  # convert all entries
# print (intt_rate)
# print (games["int_rate"])
# plt.hist(intt_rate)
# plt.show()



# games = games.dropna(axis=0)


kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = games._get_numeric_data()
# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# print 'numeric columns allllll ',games.select_dtypes(include=numerics)
print '=====',list(games.select_dtypes(include=[np.number]).columns.values)
# print 'good_columns',good_columns
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
# print 'labels',labels


# Create a PCA model.
pca_2 = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(good_columns)
# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
# Show the plot.
# plt.show()

print 'corrr.....',games.corr()["term"]   #games[games.columns[1:]].corr()["term"]

candidate_columns = list(games.select_dtypes(include=[np.number]).columns.values)
candidate_columns.remove("term")
# print candidate_columns
target = "term"

# Generate the training set.  Set random_state to be able to replicate results.
train = games.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = games.loc[~games.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)
#
# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(train[candidate_columns], train[target])

# Generate our predictions for the test set.
predictions = model.predict(test[candidate_columns])

# Compute error between our test predictions and the actual values.
mean_sqr_error = mean_squared_error(predictions, test[target])
print 'mean squr error ',mean_sqr_error
print 'mean squr error ',round(mean_sqr_error,2)

############ model 2

# Initialize the model with some parameters.
model2 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Fit the model to the data.
model2.fit(train[candidate_columns], train[target])
# Make predictions.
predictions2 = model2.predict(test[candidate_columns])
# Compute the error.
mean_sqr_error2 = mean_squared_error(predictions2, test[target])

print 'mean squr error 2 ',mean_sqr_error2
# print 'mean squr error 2',round(mean_sqr_error2,2)

############ ensemble

X = train[candidate_columns]
Y = train[target]
num_folds = 10
num_instances = len(X)
seed = 7
num_trees = 100
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())