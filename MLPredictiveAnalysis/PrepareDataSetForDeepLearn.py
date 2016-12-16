import pandas as pd
import os
import time
from datetime import datetime

from time import mktime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
style.use("dark_background")
from sklearn.preprocessing import OneHotEncoder

import re

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
        elif 'xx' in s:
            # print 'year in s ', 'years' in s
            s= s.replace('xx','')
            # print 'new s after removing years ',s

        try:
            # print 'return float of s ',s
            return float(s)
        except ValueError: # converting did not work
            # print 'exception normal s ',s
            return s  # return original string
    else:
        return s


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
            'earliest_cr_line',
            'inq_last_6mths',
            'mths_since_last_delinq',
            'open_acc',
            'total_acc',
            'out_prncp',
            'acc_now_delinq'
            ]

df = pd.DataFrame(columns = FEATURES)

data_df = pd.read_csv(os.path.join(pathTo2CsvFiles, "LoanStats_2016Q1.csv"),skiprows=1,low_memory=False,usecols=FEATURES)

valid_loan_status = ['Charged Off','Default','Fully Paid','Current']

data_df = data_df[data_df.loan_status.isin(valid_loan_status)]

data_df = data_df.applymap(f)

# print data_df

data_df = data_df.dropna(axis=0)

print data_df
print data_df.loan_status

data_df[["loan_status"]] = (data_df["loan_status"]
                               .replace("Charged Off", 0)
                               .replace("Default", 0)
                               .replace("Fully Paid", 1)
                               .replace("Current", 1)
                               )
print data_df.loan_status
print data_df

data_df[["grade"]] = (data_df["grade"]
                           .replace("A", 7)
                           .replace("B", 6)
                           .replace("C", 5)
                           .replace("D", 4)
                            .replace("E",3)
                           .replace("F", 2)
                           .replace("G", 1)
                       )




data_df[["sub_grade"]] = (data_df["sub_grade"]
                          .replace("A1", 35)
                          .replace("A2", 34)
                          .replace("A3", 33)
                          .replace("A4", 32)
                          .replace("A5", 31)
                          .replace("B1", 30)
                          .replace("B2", 29)
                            .replace("B3",28)
                           .replace("B4", 27)
                           .replace("B5", 26)
                           .replace("C1", 25)
                            .replace("C2", 24)
                           .replace("C3", 23)
                           .replace("C4", 22)
                            .replace("C5", 21)
                           .replace("D1", 20)
                           .replace("D2", 19)
                           .replace("D3", 18)
                            .replace("D4", 17)
                           .replace("D5", 16)
                           .replace("E1", 15)
                            .replace("E2", 14)
                           .replace("E3", 13)
                           .replace("E4", 12)
                           .replace("E5", 11)
                            .replace("F1", 10)
                           .replace("F2", 9)
                           .replace("F3", 8)
                            .replace("F4", 7)
                           .replace("F5", 6)
                           .replace("G1", 5)
                           .replace("G2", 4)
                            .replace("G3", 3)
                           .replace("G4", 2)
                           .replace("G5", 1)

                           )




data_df[["home_ownership"]] = (data_df["home_ownership"]
                               .replace("MORTGAGE", 1)
                               .replace("OWN", 3)
                               .replace("RENT", 2)
                           )




data_df[["verification_status"]] = (data_df["verification_status"]
                                       .replace("Not Verified", 1)
                                       .replace("Verified", 3)
                                       .replace("Source Verified", 2)
                                   )


data_df[["purpose"]] = (data_df["purpose"]
                           .replace("house", 1)
                           .replace("medical", 2)
                           .replace("major_purchase", 3)
                           .replace("moving", 4)
                            .replace("other", 5)
                            .replace("vacation", 6)
                            .replace("small_business", 7)
                            .replace("renewable_energy", 8)
                            .replace("debt_consolidation", 9)
                            .replace("credit_card", 10)
                            .replace("car", 11)
                            .replace("home_improvement", 12)
                       )

data_df[["addr_state"]] =  (data_df["addr_state"]
                            .replace('AK', 1)
                            .replace('AL', 2)
                            .replace('AZ', 3)
                            .replace('AR', 4)
                            .replace('CA', 5)
                            .replace('CO', 6)
                            .replace('CT', 7)
                            .replace('DE', 8)
                            .replace('FL', 9)
                            .replace('GA', 10)
                            .replace('HI', 11)
                            .replace('ID', 12)
                            .replace('IL', 13)
                            .replace('IN', 14)
                            .replace('IA', 15)
                            .replace('KS', 16)
                            .replace('KY', 17)
                            .replace('LA', 18)
                            .replace('ME', 19)
                            .replace('MD', 20)
                            .replace('MA', 21)
                            .replace('MI', 22)
                            .replace('MN', 23)
                            .replace('MS', 24)
                            .replace('MO', 25)
                            .replace('MT', 26)
                            .replace('NE', 27)
                            .replace('NV', 28)
                            .replace('NH', 29)
                            .replace('NJ', 30)
                            .replace('NM', 31)
                            .replace('NY', 32)
                            .replace('NC', 33)
                            .replace('ND', 34)
                            .replace('OH', 35)
                            .replace('OK', 36)
                            .replace('OR', 37)
                            .replace('PA', 38)
                            .replace('RI', 39)
                            .replace('SC', 40)
                            .replace('SD', 41)
                            .replace('TN', 42)
                            .replace('TX', 43)
                            .replace('UT', 44)
                            .replace('VT', 45)
                            .replace('VA', 46)
                            .replace('WA', 47)
                            .replace('WV', 48)
                            .replace('WI', 49)
                            .replace('WY', 50)
                            .replace('DC', 51)
                       )



data_df.to_csv(os.path.join(pathToSave, "PredictionWithHeterogeneousDataSet.csv"))



















