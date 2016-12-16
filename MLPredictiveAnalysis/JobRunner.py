# Needed packages
# pip install certifi
# pip install zope.interface
# pip install twisted pypiwin32 service_identity
# pip install pypiwin32
# pip install service_identity

# import helper
import certifi
import os
import json
from collections import OrderedDict
from simple_salesforce import Salesforce
# import pandas as pd
import predictionApi
import SelectModel
import time
from datetime import datetime
import numpy as np


os.environ["SSL_CERT_FILE"] = certifi.where()

from bayeux.bayeux_client import BayeuxClient


class JobRunner:

    def __init__(self):

        self.sf = Salesforce(username="ashish_clportal@clo.com", password="HACKERSc276",
                             security_token="XVXqD3cylx2oxt8LDeb0woAQ9", version="36.0")
        # print self.sf.session_id
        print "LoggedIn Successfully"

    def record_analysis(self, data):

        response = self.sf.query("SELECT Enable_Custom_Loan_Status_Predictor__c FROM MySetting__c")

        if response['records'][0]['Enable_Custom_Loan_Status_Predictor__c']:
            self.internal_api(data)
        else:
            self.external_api(data)

    def internal_api(self,data):
        pass

    def external_api(self, data):

        fields = [
                "Loan_Amount__c",
                "Term__c","Interest_Rate__c",
                "Grade__c",
                "Sub_Grade__c",
                "Employee_Length__c",
                "Home_Ownership__c",
                "Annual_Income__c",
                "Verification_Status__c",
                "Purpose__c",
                "Zip_Code__c",
                "Address_State__c",
                "Dti__c",
                "Delinquent_2yrs__c",
                "Earliest_Credit_Line__c",
                "Inquiry_Last_6mths__c",
                "Months_Since_Last_Delinquent__c",
                "Open_Account__c",
                "Total_Account__c",
                "Out_Principle__c",
                "Account_Now_Delinquent__c"
                ]

        field_values = data["data"]["sobject"]
        # print field_values
        valuesToSend = []
        listOfFields = []
        updateId = data["data"]["sobject"]["Id"]
        # print 'updateId ', updateId

        for value in fields:
            if value in data["data"]["sobject"] and data["data"]["sobject"][value] is not None:
                if value == 'Earliest_Credit_Line__c' and data["data"]["sobject"][value]:
                    dt64 = np.datetime64(data["data"]["sobject"][value])
                    x = str(dt64).index('T')
                    dt64 = str(dt64)[:x]
                    # print dt64
                    unix = time.mktime(datetime.strptime(dt64, "%Y-%m-%d").timetuple())
                    formattedTime = datetime.fromtimestamp(int(unix)).strftime('%B-%y')
                    # print formattedTime
                    valuesToSend.append(formattedTime)
                else:
                    valuesToSend.append(data["data"]["sobject"][value])

                listOfFields.append(value)

        # print 'listToSend ',listToSend
        # print 'fieldsList', listOfFields

        # predictionApi.fields_list = listOfFields
        predictionApi.model_id, predictionApi.file_name, predictionApi.listToSend = SelectModel.model(listOfFields,
                                                                                                      valuesToSend)
        # print "model", predictionApi.model_id
        loan_status_response_prediction = predictionApi.main()
        # print 'loan_status_response_prediction ',loan_status_response_prediction

        # write it to SF object
        self.sf.Loan_Application_Prediction__c.update(updateId, {'Predicted_Loan_Status__c': loan_status_response_prediction})


        # predictionApi.main() --
        #save the record Id for later update --
        #convert data to list to send as csv instance --
        #call the ML class function with argument as list
        # store the response loan status token
        #write it back to Sf record

    def runJob(self, query):
        if(query == None) :
            return False

        version = str(36.0)

        name = 'LoanStatusPrediction'   #'Regex'
        # sdata = {'Name':  name,
        #          'Query': query,
        #          'ApiVersion': version,
        #          'NotifyForOperationCreate': 'true',
        #          'NotifyForFields': 'Referenced'
        #          }
        # topicId = self.create_record('PushTopic', sdata, self.sf)['id']
        # print topicId
        url = 'https://' + self.sf.sf_instance.encode()
        # print 'base url : ' + url
        client = BayeuxClient(url + '/cometd/' + version, 'OAuth ' + self.sf.session_id)
        # print "737373"
        client.register('/topic/' + name, self.record_analysis)
        # client.register('/chat/demo', cb)
        try:
            client.start()

            while True:
                pass
            # self.delete_record('PushTopic', topicId)

        except Exception as e:
            print str(e)

    def create_record(self, sobj, sdata, sf):

        new_url = self.sf.base_url + 'sobjects/{object_name}/'.format(object_name=sobj)
        response = self.sf._call_salesforce('POST', new_url, data=json.dumps(sdata))

        return response.json(object_pairs_hook=OrderedDict)

    def delete_record(self, sobj, sid):
        new_url = self.sf.base_url + 'sobjects/{object_name}/{object_id}/'.format(object_name=sobj, object_id=sid)
        response = self.sf._call_salesforce('DELETE', new_url)

        return response


if __name__ == "__main__":
    s = JobRunner()
    # query = "SELECT Id, Name, Api_Name__c, Regular_Expression__c FROM Api_Regular_Expression__c"
    s.runJob(query="")



