import httplib2, argparse, os, sys, json
from oauth2client import tools, file, client
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient import discovery
from googleapiclient.errors import HttpError
import SelectModel

# Project and model configuration
project_id = 'loanstatusprediction'
model_id = ''       # status-prediction-model
file_name = ''      # googlePredictionToUpload

# activity labels
labels = {
    '1':'Charged Off',
    '2':'Default',
    '3':'Fully Paid',
    '4':'Current'
}

listToSend = []     # list of values for which prediction should be done
fields_list = []    # list of fields according to which predictor model is selected

model_id, file_name, listToSend = SelectModel.model(fields_list, listToSend)

def main():
    """ Simple logic: train and make prediction """
    try:
        # print 'listToSend in prediction api',listToSend
        prediction = make_prediction()
        # print 'prediction before return ' + prediction
        return prediction
    except HttpError as e:
        if e.resp.status == 404:  # model does not exist
            print("Model does not exist yet.")
            train_model()
            try:
                prediction = make_prediction()
                # print 'prediction before return '+prediction
                return prediction
            except Exception as e:
                print 'exception ....',str(e)
        else:  # real error
            print 'exception ',str(e)
            print(e)


def make_prediction():
    """ Use trained model to generate a new prediction """

    api = get_prediction_api()

    # print("Fetching model.")

    model = api.trainedmodels().get(project=project_id, id=model_id).execute()

    if model.get('trainingStatus') != 'DONE':
        print("Model is (still) training. \nPlease wait and run me again!")  # no polling
        try:
            exit()
        except SystemExit,e:
            os._exit(1)

    # print("Model is ready.")
    # pathToSave = "A:\CSV"
    print " "
    print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" \
          "^"
    """
    #Optionally analyze model stats (big json!)
    analysis = api.trainedmodels().analyze(project=project_id, id=model_id).execute()
    print(analysis)
    exit()
    """

    # read new record from local file
    # with open(os.path.join(pathToSave, "recordForTest.csv")) as f:
    #     record = f.readline().split(',')  # csv
    #     # print 'record',record
    # print 'csvInstance listToSend ',listToSend

    prediction = api.trainedmodels().predict(project=project_id, id=model_id, body={
        'input': {
            'csvInstance': listToSend
        },
    }).execute()

    # retrieve classified label and reliability measures for each class
    # print 'prediction ',prediction
    label = prediction.get('outputLabel')
    stats = prediction.get('outputMulti')

    # print 'Predicted Loan Status is : ', label
    print "--------   Probabilities for each status   --------"
    for i in stats:
        print i['label'], (' ')*(11 - len(i['label']))+': ', i['score']
    print "--------   Predicted Status   ---------------------"
    print label
    return label


def train_model():
    """ Create new classification model """

    api = get_prediction_api()

    print("Creating new Model.")

    api.trainedmodels().insert(project=project_id, body={
        'id': model_id,
        'storageDataLocation': 'loanstatusprediction/'+file_name+'.csv',
        'modelType': 'CLASSIFICATION'
    }).execute()


def get_prediction_api(service_account=True):
    scope = [
        'https://www.googleapis.com/auth/prediction',
        'https://www.googleapis.com/auth/devstorage.read_only'
    ]
    return get_api('prediction', scope, service_account)


def get_api(api, scope, service_account=True):
    """ Build API client based on oAuth2 authentication """
    STORAGE = file.Storage("InputFiles/credentials/LoanStatusPrediction-52106d4c493a.json")  # local storage of oAuth tokens
    # credentials = STORAGE.get()
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        "InputFiles/credentials/LoanStatusPrediction-52106d4c493a.json", scopes=scope)
    if credentials is None or credentials.invalid:  # check if new oAuth flow is needed
        if service_account:  # server 2 server flow
            # credentials = ServiceAccountCredentials('C:/Users/lenovo/Google Drive/Laptop Saved/LoanStatusPrediction-52106d4c493a.json', scopes=scope)
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                    'C:/Users/Lenovo/Google Drive/Laptop Saved/LoanStatusPrediction-52106d4c493a.json', scopes=scope)
            STORAGE.put(credentials)
        else:  # normal oAuth2 flow
            # CLIENT_SECRETS = os.path.join(os.path.dirname(__file__), 'client_secrets.json')
            # FLOW = client.flow_from_clientsecrets(CLIENT_SECRETS, scope=scope)
            # PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
            #                                  parents=[tools.argparser])
            # FLAGS = PARSER.parse_args(sys.argv[1:])
            # credentials = tools.run_flow(FLOW, STORAGE, FLAGS)
            print 'could not authenticate'

    # wrap http with credentials
    http = credentials.authorize(httplib2.Http())
    return discovery.build(api, "v1.6", http=http)


if __name__ == '__main__':
    main()