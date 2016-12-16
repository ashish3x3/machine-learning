import glob
import pandas as pd
import os
import time
from datetime import datetime
import numpy as np
import io

path = "C:\Users\lenovo\Downloads\Compressed\LoanStats3a.csv_2\csv"
pathToFiles = "C:\Users\lenovo\Downloads\Compressed\DataSet\CsvDataFiles"
pathTo2CsvFiles = "C:\Users\lenovo\Downloads\Compressed\DataSet\CsvDataFiles\Twocsvfiles"
pathToSave = "A:\CSV"
class MergeCsvFiles:
    def MergeCsv(self):
        allFiles = glob.glob(os.path.join(pathTo2CsvFiles, "*.csv"))
        print 'allFiles',allFiles

        frame = pd.DataFrame()
        list_ = []

        # fieldsToReorder = ['loan_status','loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 'emp_length',
        #           'home_ownership','annual_inc','verification_status','purpose','zip_code','addr_state',
        #           'dti','delinq_2yrs','earliest_cr_line','inq_last_6mths','mths_since_last_delinq','open_acc',
        #           'total_acc','out_prncp','acc_now_delinq'
        #           ]
        #
        # df = pd.read_csv(os.path.join(pathToSave, "merge12.csv"))
        # df_reorder = df[fieldsToReorder]  # rearrange column here
        # df_reorder.to_csv(os.path.join(pathToSave, "merge13.csv"),index=False)

        for file_ in allFiles:
            print 'file_ ######### ',file_
            # df = pd.DataFrame(columns=['loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 'emp_length','home_ownership',
            #                            'annual_inc','verification_status','purpose','zip_code','addr_state','dti','delinq_2yrs',
            #                            'earliest_cr_line','inq_last_6mths','mths_since_last_delinq','open_acc','total_acc',
            #                            'out_prncp','acc_now_delinq','loan_status'
            #                            ])
            # merge_df = pd.DataFrame.from_csv(file_)
            # print merge_df
            # fileToSave = glob.glob(os.path.join(path, "merge.csv"))
            # print 'filrToSave #### ', fileToSave
            np_array_list = []

            fields = ['loan_amnt','term', 'int_rate', 'grade', 'sub_grade', 'emp_length','home_ownership',
                       'annual_inc','verification_status','purpose','zip_code','addr_state','dti','delinq_2yrs',
                       'earliest_cr_line','inq_last_6mths','mths_since_last_delinq','open_acc','total_acc',
                       'out_prncp','acc_now_delinq','loan_status'
                    ]
            # df = pd.read_csv(file_, skipinitialspace=True, names=fields, low_memory=False)
            df = pd.read_csv(file_, usecols=fields,skiprows=1,header=0,index_col=None)
            list_.append(df)

            # print 'df.keys########',df.keys()
            # print 'df @@@@@', df
            # print 'df.as_matrix() ####', df.as_matrix(columns="fields")
            # np_array_list.append(df.as_matrix())
            # print 'np_array_list() ####', np_array_list
            # comb_np_array = np.vstack(np_array_list)
            # print 'comb_np_array ####',comb_np_array
            # big_frame = pd.DataFrame(comb_np_array)
            # big_frame.columns = fields
            # # print 'big_frame#### ', big_frame
            # big_frame.to_csv(os.path.join(pathToSave, "merge2.csv"))

            # See the keys
            # print 'df.keys########',df.keys()
            # print 'df @@@@@', df
            #
            # frame = pd.DataFrame()
            # list_ = []
            #
            # list_.append(df)
            # frame = pd.concat(list_)
            # # frame.columns = fields
            # print 'frame#### ',frame
            #
            # frame.to_csv(os.path.join(pathToSave, "merge7.csv"))

        frame = pd.concat(list_,ignore_index=True)
        # frame.drop(frame.columns[[0]], axis=1,inplace=True)
        frame.to_csv(os.path.join(pathToSave, "merge15.csv"),index=False)
        # df[df.columns].to_csv()


if __name__ == "__main__":
    s =  MergeCsvFiles()
    s.MergeCsv()