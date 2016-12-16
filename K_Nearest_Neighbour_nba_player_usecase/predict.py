import pandas
import os
import settings
import math
from scipy.spatial import distance
import random
from numpy.random import permutation
from sklearn.neighbors import KNeighborsRegressor


with open(os.path.join(settings.pathToCsv,"nba_2013.csv"), 'r') as csvfile:
    nba = pandas.read_csv(csvfile)

print 'nba with index ',nba.head(5)
# The names of all the columns in the data.
print(nba.columns.values),nba.shape


# Select Lebron James from our dataset
selected_player = nba[nba["player"] == "LeBron James"].iloc[0]

# Choose only the numeric columns (we'll use these to compute euclidean distance)
distance_columns = ['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts']

def euclidean_distance(row):
    """
    A simple euclidean distance function
    """
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - selected_player[k]) ** 2
    return math.sqrt(inner_value)

# Find the distance from each player in the dataset to lebron.
lebron_distance = nba.apply(euclidean_distance, axis=1)

# print 'lebron_distance ',lebron_distance
print 'nba ',nba.head(5)


print(nba.columns.values),nba.shape

# Select only the numeric columns from the NBA dataset
nba_numeric = nba[distance_columns]

# Normalize all of the numeric columns
nba_normalized = (nba_numeric - nba_numeric.mean()) / nba_numeric.std()
# print 'nba_normalized ',nba_normalized.head(5)


# Fill in NA values in nba_normalized
nba_normalized.fillna(0, inplace=True)

#=========================
#
# # Find the normalized vector for lebron james.
# lebron_normalized = nba_normalized[nba["player"] == "LeBron James"]
# print 'lebron_normalized ',lebron_normalized
#
# # Find the distance between lebron james and everyone else.
# euclidean_distances = nba_normalized.apply(lambda row: distance.euclidean(row, lebron_normalized), axis=1)
# print 'euclidean_distances ',euclidean_distances
#
# # Create a new dataframe with distances.
# distance_frame = pandas.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
# distance_frame.sort("dist", inplace=True)
# print 'distance_frame ',distance_frame
# # Find the most similar player to lebron (the lowest distance to lebron is lebron, the second smallest is the most similar non-lebron player)
# #get the index of 1st row
# second_smallest = distance_frame.iloc[1]["idx"]
# #get the player name by the above index from nba
# most_similar_to_lebron = nba.loc[int(second_smallest)]["player"]  #using loc as it works with label index i.e player here
# print 'most_similar_to_lebron ',most_similar_to_lebron

#==============
nba = nba.dropna(axis=0, how='any')
# Randomly shuffle the index of nba.
random_indices = permutation(nba.index)
# print 'random_indices ',random_indices
# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
test_cutoff = math.floor(len(nba)/3)
# print 'test_cutoff ',test_cutoff
# Generate the test set by taking the first 1/3 of the randomly shuffled indices.

test = nba.loc[random_indices[1:test_cutoff]]
# print 'nba.loc[random_indices[1:test_cutoff]] ',nba.loc[random_indices[1:test_cutoff]]
print 'test ',test.iloc[:5]
# print 'test 1 ',test.iloc[:1]
# Generate the train set with the rest of the data.
train = nba.loc[random_indices[test_cutoff:]]
print 'train ',train.iloc[:5]

# The columns that we will be making predictions with.
x_columns = ['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf']
# The column that we want to predict.
y_column = ["pts"]

# Create the knn model.
# Look at the five closest neighbors.
knn = KNeighborsRegressor(n_neighbors=5)
# Fit the model on the training data.
knn.fit(train[x_columns], train[y_column])
# Make point predictions on the test set using the fit model.
predictions = knn.predict(test[x_columns])

toTest = [[34, 78, 8,  1726, 150, 413,  0.363, 63,  163, 0.386503067, 87,  250, 0.348,       0.439, 40,  52,  0.769, 20, 143, 163, 147, 47, 19, 54, 124],
          [35, 80, 80, 2628, 633, 1273, 0.497, 131, 329, 0.398176292, 502, 944, 0.531779661, 0.549, 338, 376, 0.899, 40, 458, 498, 216, 73, 45, 117, 165]
        ] # 403 1735
predictionsRealTime = knn.predict(toTest)
print 'predictionsRealTime ',predictionsRealTime

# Get the actual values for the test set.
actual = test[y_column]

print 'predictions ',predictions,test[x_columns].index
print 'actual ',actual

# Compute the mean squared error of our predictions.
mse = (((predictions - actual) ** 2).sum()) / len(predictions)
print 'mse ',mse
rmse = math.sqrt(mse)
print 'rmse ',rmse
