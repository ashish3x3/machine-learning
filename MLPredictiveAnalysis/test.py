from simple_salesforce import Salesforce


sf = Salesforce(username="ashish_clportal@clo.com", password="HACKERSc276",
                     security_token="XVXqD3cylx2oxt8LDeb0woAQ9", version="36.0")

response = sf.query("SELECT Enable_Custom_Loan_Status_Predictor__c FROM MySetting__c")

print type(response['records'][0]['Enable_Custom_Loan_Status_Predictor__c'])