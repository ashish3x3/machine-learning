

all_categories = {'Loan_Amount__c': '01',
                  'Term__c': '02',
                  'Interest_Rate__c': '03',
                  'Grade__c': '04',
                  'Sub_Grade__c': '05',
                  'Employee_Length__c': '06',
                  'Home_Ownership__c': '07',
                  'Annual_Income__c': '08',
                  'Verification_Status__c': '09',
                  'Purpose__c': '10',
                  'Zip_Code__c': '11',
                  'Address_State__c': '12',
                  'Dti__c': '13',
                  'Delinquent_2yrs__c': '14',
                  'Earliest_Credit_Line__c': '15',
                  'Inquiry_Last_6mths__c': '16',
                  'Months_Since_Last_Delinquent__c': '17',
                  'Open_Account__c': '18',
                  'Total_Account__c': '19',
                  'Out_Principle__c': '20',
                  'Account_Now_Delinquent__c': '21'}

borrower_category = {'Employee_Length__c': '06',
                     'Home_Ownership__c': '07',
                     'Annual_Income__c': '08',
                     'Zip_Code__c': '11',
                     'Address_State__c': '12'}

loan_category = {'Loan_Amount__c': '01',
                 'Term__c': '02',
                 'Interest_Rate__c': '03',
                 'Verification_Status__c': '09',
                 'Purpose__c': '10',
                 'Out_Principle__c': '20',
                 'Account_Now_Delinquent__c': '21'}

credit_category = {'Grade__c': '04',
                   'Sub_Grade__c': '05',
                   'Dti__c': '13',
                   'Delinquent_2yrs__c': '14',
                   'Earliest_Credit_Line__c': '15',
                   'Inquiry_Last_6mths__c': '16',
                   'Months_Since_Last_Delinquent__c': '17',
                   'Open_Account__c': '18',
                   'Total_Account__c': '19'}

# lst = ['Loan_Amount__c', 'Term__c', 'Interest_Rate__c', 'Verification_Status__c', 'Purpose__c',
#        'Out_Principle__c', 'Account_Now_Delinquent__c']


def model(field_lst, value_list):
    if len(field_lst) != len(all_categories):
        b_count = 0
        b_list = []
        l_count = 0
        l_list = []
        c_count = 0
        c_list = []
        for f in field_lst:
            if f in borrower_category:
                b_count += 1
                b_list.append(value_list[field_lst.index(f)])
            if f in loan_category:
                l_count += 1
                l_list.append(value_list[field_lst.index(f)])
            if f in credit_category:
                c_count += 1
                c_list.append(value_list[field_lst.index(f)])

        if b_count == 5:
            return 'b_model', "borrower_characteristics_for_upload_no_headers", b_list
        elif l_count == 7:
            return 'l_model', "loan_characteristics_for_upload_no_headers", l_list
        elif c_count == 9:
            return 'c_model', "credit_characteristics_for_upload_no_headers", c_list
        else:
            return '', '', ''
    else:
        return "status-prediction-model", "googlePredictionToUpload", value_list
