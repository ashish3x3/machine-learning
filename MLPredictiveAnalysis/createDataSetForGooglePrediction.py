import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import statistics
import os
import time
from datetime import datetime
import re
import glob

style.use("dark_background")
#
# path = "C:\Users\lenovo\Downloads\Compressed\LoanStats3a.csv_2\csv"
# pathToFiles = "C:\Users\lenovo\Downloads\Compressed\DataSet\CsvDataFiles"
# pathTo2CsvFiles = "C:\Users\lenovo\Downloads\Compressed\DataSet\CsvDataFiles\Twocsvfiles"
# pathToSave = "A:\CSV"
#
# def f(s):
#     """ convert string to float if possible """
#     if isinstance(s, str):
#         # print 's is',s,len(s)
#         s = s.strip()   # remove spaces at beginning and end of string
#         # print 's after strip is',s, len(s)
#         # print 'months in s ','months' in s
#
#         if s.endswith('%'):  # remove %, if exists
#             # print 's ends with %',s,len(s)
#             s = s[:-1]
#         elif 'months' in s:
#             s= s.replace('months','').replace('+','').replace('>','')
#             # print 'new s after removing months ',s
#         elif 'years' in s:
#             # print 'years in s ', 'years' in s
#             s= s.replace('years','').replace('year','').replace('+','').replace('>','')
#             # print 'new s after removing years ',s
#         elif 'year' in s:
#             # print 'year in s ', 'years' in s
#             s= s.replace('year','').replace('+','').replace('>','').replace('<','')
#             # print 'new s after removing years ',s
#
#         try:
#             # print 'return float of s ',s
#             return float(s)
#         except ValueError: # converting did not work
#             # print 'exception normal s ',s
#             return s  # return original string
#     else:
#         return s
#
# ################ create
# #
# # fileToRead = glob.glob(os.path.join(pathTo2CsvFiles, "LoanStats_2016Q1.csv"))
# #
# # frame = pd.DataFrame()
# # list_ = []
# #
# # fields = ['loan_amnt','term', 'int_rate', 'grade', 'sub_grade', 'emp_length','home_ownership',
# #            'annual_inc','verification_status','purpose','zip_code','addr_state','dti','delinq_2yrs',
# #            'earliest_cr_line','inq_last_6mths','mths_since_last_delinq','open_acc','total_acc',
# #            'out_prncp','acc_now_delinq','loan_status'
# #         ]
# # # df = pd.read_csv(file_, skipinitialspace=True, names=fields, low_memory=False)
# # df = pd.read_csv(fileToRead[0], usecols=fields,skiprows=1,header=0,index_col=None)
# # list_.append(df)
# # frame = pd.concat(list_,ignore_index=True)
# # # frame.drop(frame.columns[[0]], axis=1,inplace=True)
# # frame.to_csv(os.path.join(pathToSave, "googlePrediction.csv"),index=False)
#
#
# ########## reorder column, remove unnecesary loan status,and other unrequired rows..
#
# fieldsToReorder = ['loan_status','loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 'emp_length',
#                   'home_ownership','annual_inc','verification_status','purpose','zip_code','addr_state',
#                   'dti','delinq_2yrs','earliest_cr_line','inq_last_6mths','mths_since_last_delinq','open_acc',
#                   'total_acc','out_prncp','acc_now_delinq'
#                   ]
#
# df = pd.read_csv(os.path.join(pathToSave, "googlePrediction.csv"))
# valid_loan_status = ['Charged Off','Default','Fully Paid','Current']
# df = df[df.loan_status.isin(valid_loan_status)]
# df = df.dropna(axis=0)
# df = df.applymap(f)
# df_reorder = df[fieldsToReorder]  # rearrange column here
# df_reorder.to_csv(os.path.join(pathToSave, "googlePredictionToUpload.csv"),index=False)
#
# import date_converter

dt64 = np.datetime64('2002-06-28T01:00:00.000000000+0100')
x=  str(dt64).index('T')
dt64 = str(dt64)[:x]
print dt64
unix = time.mktime(datetime.strptime(dt64, "%Y-%m-%d").timetuple())
formattedTime  = datetime.fromtimestamp(int(unix)).strftime('%B-%y')
print formattedTime