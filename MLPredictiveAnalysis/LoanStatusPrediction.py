# from google.auth import compute_engine
# from google.cloud import datastore
from oauth2client.client import GoogleCredentials
# from oauth2client.contrib.appengine import AppAssertionCredentials
# from oauth2client.service_account import ServiceAccountCredentials
from httplib2 import Http
from oauth2client.service_account import ServiceAccountCredentials
from apiclient.discovery import build

from google.cloud import storage

from googleapiclient import discovery


# Instantiates a client
# storage_client = storage.Client()

# The name for the new bucket
bucket_name = 'my-new-bucket'

# client = Client.from_service_account_json('C:/Users/lenovo/Google Drive/Laptop Saved/LoanStatusPrediction-52106d4c493a.json')

credentials = GoogleCredentials.get_application_default()
compute = discovery.build('compute', 'v1', credentials=credentials)

# scopes = ['https://www.googleapis.com/auth/sqlservice.admin']
#
# credentials = ServiceAccountCredentials.from_json_keyfile_name(
#     'C:/Users/lenovo/Google Drive/Laptop Saved/LoanStatusPrediction-52106d4c493a.json', scopes=scopes)
# http_auth = credentials.authorize(Http())
#
# sqladmin = build('sqladmin', 'v1beta3', http=http_auth)