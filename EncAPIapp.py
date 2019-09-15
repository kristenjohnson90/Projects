import json
import requests
import pandas as pd
from pandas.io.json import json_normalize
import time
import config

# Start timer
start_time = time.time()

pd.set_option('display.max_columns', None)
desired_width = 170
pd.set_option('display.width', desired_width)

client_id = config.client_id
client_secret = config.client_secret
token_url = config.token_url
username = config.username
password = config.password

data = {'grant_type': 'password', 'username': username, 'password': password}

# send request to get token
access_token_response = requests.post(token_url, data=data, allow_redirects=False, auth=(client_id, client_secret))
token_body = json.loads(access_token_response.text)

# access token
token = token_body['access_token']
print('access token: ' + token)

# create empty list
enc_list = []

# Get specific fields and filter by
filter = {"filter": {
    "operator": "or",
    "terms": [
        {
            "operator": "and",
            "terms": [
                {"canonicalName": "Fields.1999", "value": "09/01/2019", "matchType": "greaterThanOrEquals", "precision":"day"},
                {"canonicalName": "Fields.1999", "value": "09/30/2019", "matchType": "lessThanOrEquals", "precision":"day"}
        ]},
        {
            "operator": "and",
            "terms": [
                {"canonicalName": "Fields.CX.BROK.FUNDED", "value": "09/01/2019", "matchType": "greaterThanOrEquals", "precision":"day"},
                {"canonicalName": "Fields.CX.BROK.FUNDED", "value": "09/30/2019", "matchType": "lessThanOrEquals", "precision":"day"}
            ]}
    ]},

 "fields": [
     "Fields.1999",
     "Fields.364",
     "Fields.CX.BROK.FUNDED",
     "Fields.CX.ORIG.DESC.ROLLUP",
     "loan.GUID",
     "Loan.LastModified"
 ]
}

# max limit
url = "https://api.elliemae.com/encompass/v1/loanPipeline?cursorType=randomAccess&limit=25000&start=0"

# Send api request for data from ellie mae
api_request = requests.post(url=url, json=filter, headers={'Authorization': 'Bearer ' + token})

# Print response code
# print(api_request)

# Convert to json and DataFrame
json_file = api_request.json()
df = json_normalize(json_file)

print("First matrix:" + str(df.shape))
limit = str(df.shape[0])

# Get cursor
cursor = str(api_request.headers.get('x-cursor'))
# print(cursor)

# Add df to enc_list list
enc_list.append(df)

# Get specific fields for 2nd call and on
filter = {"fields": [
     "Fields.1999",
     "Fields.364",
     "Fields.CX.BROK.FUNDED",
     "Fields.CX.ORIG.DESC.ROLLUP",
     "loan.GUID",
     "Loan.LastModified"
]
}

# Start for loop collecting ellie mae data
for i in range(1, 100):
    print("iteration = " + str(i))

    # new start each run
    start = str(int(limit) * i)
    print("start value = " + start)

    url = "https://api.elliemae.com/encompass/v1/loanPipeline?cursor=" + cursor + "&limit=" + limit + "&start=" + start

    # Send api request for data from ellie mae
    # api_request = requests.post(url=url, json=dict(a=None, b=1), headers={'Authorization':'Bearer ' + token})
    api_request = requests.post(url=url, json=filter, headers={'Authorization': 'Bearer ' + token})

    # Print response code
    # print(api_request)

    # Convert to json and DataFrame
    json_file = api_request.json()
    df = json_normalize(json_file)

    # print(df.shape)
    if df.shape[0] == 1:
        break

    print("Matrix: " + str(df.shape))
    enc_list.append(df)
    # End For Loop


# Combine list into one DataFrame
df = pd.concat(enc_list, sort=True)
print(df.shape)

# print(df.columns)


headers = ["funds_released_date", "loan_number", "brokered_funded_date", "division", "guid", "last_modified"]

# Rename column names
df.columns = headers

# print(df.dtypes)
print(df.describe())
# print(df[:].head())

# Export as CSV
print(df.shape)
# df.to_csv("enc_data.csv")
df.to_csv("enc_data_funded.csv")
print("--- %s seconds ---" % (time.time() - start_time))

