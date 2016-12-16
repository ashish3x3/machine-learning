import pandas as pd
import os

pathToCsv = "A:\CSV\Titanic Case Study"

# We can use the pandas library in python to read in the csv file.
# This creates a pandas dataframe and assigns it to the titanic variable.
titanic = pd.read_csv(os.path.join(pathToCsv, "train.csv"))

# Print the first 5 rows of the dataframe.
print(titanic.head(5))
print(titanic.describe())

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Find all the unique values for "Embarked".
print(titanic["Embarked"].unique())
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

print(titanic.head(5))

#Linear regression follows the equation y=mx+by=mx+b, where y is the value we're trying to predict,
# m is a coefficient called the slope, x is the value of a column, and b is a constant called the intercept.


