import json
import requests
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
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
    "operator": "and",
    "terms": [
        {"canonicalName": "Fields.MS.START", "value": "01/01/2012", "matchType": "greaterThanOrEquals", "precision": "day"},
        {"canonicalName": "Fields.MS.START", "value": "07/31/2019", "matchType": "lessThanOrEquals", "precision": "day"},
        {
            "operator": "or",
            "terms": [
            {"canonicalName": "Fields.LOANFOLDER", "value": "Closed", "matchType": "exact"},
            {"canonicalName": "Fields.LOANFOLDER", "value": "Employee Closed", "matchType": "exact"}
            #{"canonicalName": "Fields.LOANFOLDER", "value": "Employee Adverse", "matchType": "exact"},
            #{"canonicalName": "Fields.LOANFOLDER", "value": "Adverse", "matchType": "exact"},
            #{"canonicalName": "Fields.LOANFOLDER", "value": "Scenario PreQual Archive", "matchType": "exact"}
        ]}


# --------------------------------------- ALL ACTIVE LOANS -----------------------------------------------------------
#    "operator": "and",
#    "terms": [
#        {"canonicalName": "Fields.MS.START", "value": "07/01/2019", "matchType": "isNotEmpty"},
#        {"canonicalName": "Fields.1999", "value": "07/01/2019", "matchType": "isEmpty"},
#        {"canonicalName": "Fields.CX.BROK.FUNDED", "value": "07/01/2019", "matchType": "isEmpty"},
#        {
#            "operator": "or",
#            "terms": [
#            {"canonicalName": "Fields.LOANFOLDER", "value": "My Pipeline", "matchType": "exact"},
#            {"canonicalName": "Fields.LOANFOLDER", "value": "Employee Loans", "matchType": "exact"},
#            {"canonicalName": "Fields.LOANFOLDER", "value": "Prospects", "matchType": "exact"}
#        ]}


        # {
        #     "operator": "and",
        #     "terms": [
        #     {"canonicalName": "Fields.LOANFOLDER", "value": "Trash", "matchType": "exact", "include": "false"},
        #     {"canonicalName": "Fields.LOANFOLDER", "value": "(Archive)", "matchType": "exact", "include": "false"},
        #     {"canonicalName": "Fields.LOANFOLDER", "value": "Test", "matchType": "exact", "include": "false"}
        # ]}


    ]},

 "fields": [
     "Fields.1041",
     "Fields.11",
     "Fields.1109",
     "Fields.1172",
     "Fields.12",
     "Fields.1389",
     "Fields.1395",
     "Fields.1396",
     "Fields.14",
     "Fields.1401",
     "Fields.1402",
     "Fields.1416",
     "Fields.1417",
     "Fields.1418",
     "Fields.1419",
     "Fields.15",
     "Fields.1524",
     "Fields.1525",
     "Fields.1526",
     "Fields.1527",
     "Fields.1528",
     "Fields.1529",
     "Fields.1530",
     "Fields.1532",
     "Fields.1533",
     "Fields.1534",
     "Fields.1535",
     "Fields.1536",
     "Fields.1537",
     "Fields.1538",
     "Fields.16",
     "Fields.1771",
     "Fields.1785",
     "Fields.18",
     "Fields.1811",
     "Fields.1821",
     "Fields.19",
     "Fields.1999",
     "Fields.1997",
     "Fields.2353",
     "Fields.2370",
     "Fields.2626",
     "Fields.3",
     "Fields.3142",
     "Fields.317",
     "Fields.353",
     "Fields.356",
     "Fields.364",
     "Fields.4",
     "Fields.4144",
     "Fields.4145",
     "Fields.4146",
     "Fields.4147",
     "Fields.4148",
     "Fields.4149",
     "Fields.4150",
     "Fields.4151",
     "Fields.4152",
     "Fields.4153",
     "Fields.4154",
     "Fields.4155",
     "Fields.4156",
     "Fields.4157",
     "Fields.4158",
     "Fields.4193",
     "Fields.4194",
     "Fields.420",
     "Fields.4205",
     "Fields.4210",
     "Fields.4211",
     "Fields.471",
     "Fields.478",
     "Fields.52",
     "Fields.608",
     "Fields.699",
     "Fields.734",
     "Fields.740",
     "Fields.742",
     "Fields.799",
     "Fields.84",
     "Fields.CX.BROK.FUNDED",
     "Fields.CX.ORIG.BRANCH",
     "Fields.CX.ORIG.DESC.ROLLUP",
     "Fields.CX.ORIG.ROC2",
     "Fields.CX.PER.EXCREQ",
     "Fields.CX.PER.SECREV",
     "Fields.CX.REFERRALTYPE",
     "Fields.FR0107",
     "Fields.FE0116",
     "Fields.LOANFOLDER",
     "Fields.Log.MS.Date.Approval",
     "Fields.Log.MS.Date.Docs Signing",
     "Fields.Log.MS.Date.Processing",
     "Fields.MS.START",
     "Fields.VASUMM.X23",
     "loan.GUID",
     "Loan.LastModified"
 ],
"sortOrder": [{"canonicalName": "Fields.MS.START", "order": "desc"}]
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
    "Fields.1041",
    "Fields.11",
    "Fields.1109",
    "Fields.1172",
    "Fields.12",
    "Fields.1389",
    "Fields.1395",
    "Fields.1396",
    "Fields.14",
    "Fields.1401",
    "Fields.1402",
    "Fields.1416",
    "Fields.1417",
    "Fields.1418",
    "Fields.1419",
    "Fields.15",
    "Fields.1524",
    "Fields.1525",
    "Fields.1526",
    "Fields.1527",
    "Fields.1528",
    "Fields.1529",
    "Fields.1530",
    "Fields.1532",
    "Fields.1533",
    "Fields.1534",
    "Fields.1535",
    "Fields.1536",
    "Fields.1537",
    "Fields.1538",
    "Fields.16",
    "Fields.1771",
    "Fields.1785",
    "Fields.18",
    "Fields.1811",
    "Fields.1821",
    "Fields.19",
    "Fields.1999",
    "Fields.1997",
    "Fields.2353",
    "Fields.2370",
    "Fields.2626",
    "Fields.3",
    "Fields.3142",
    "Fields.317",
    "Fields.353",
    "Fields.356",
    "Fields.364",
    "Fields.4",
    "Fields.4144",
    "Fields.4145",
    "Fields.4146",
    "Fields.4147",
    "Fields.4148",
    "Fields.4149",
    "Fields.4150",
    "Fields.4151",
    "Fields.4152",
    "Fields.4153",
    "Fields.4154",
    "Fields.4155",
    "Fields.4156",
    "Fields.4157",
    "Fields.4158",
    "Fields.4193",
    "Fields.4194",
    "Fields.420",
    "Fields.4205",
    "Fields.4210",
    "Fields.4211",
    "Fields.471",
    "Fields.478",
    "Fields.52",
    "Fields.608",
    "Fields.699",
    "Fields.734",
    "Fields.740",
    "Fields.742",
    "Fields.799",
    "Fields.84",
    "Fields.CX.BROK.FUNDED",
    "Fields.CX.ORIG.BRANCH",
    "Fields.CX.ORIG.DESC.ROLLUP",
    "Fields.CX.ORIG.ROC2",
    "Fields.CX.PER.EXCREQ",
    "Fields.CX.PER.SECREV",
    "Fields.CX.REFERRALTYPE",
    "Fields.FR0107",
    "Fields.FE0116",
    "Fields.LOANFOLDER",
    "Fields.Log.MS.Date.Approval",
    "Fields.Log.MS.Date.Docs Signing",
    "Fields.Log.MS.Date.Processing",
    "Fields.MS.START",
    "Fields.VASUMM.X23",
    "loan.GUID",
    "Loan.LastModified"

],
    "sortOrder": [{"canonicalName": "Fields.MS.START", "order": "desc"}]
}


# Start for loop collecting ellie mae data
for i in range(1,300):
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


headers = ["subject_property_type", "subject_property_address", "loan_amount", "loan_type", "subject_property_city",
           "total_income", "subject_property_state_code", "subject_property_county_code", "subject_property_state",
           "loan_program", "borr_dob", "borr_mailing_address", "borr_mailing_city", "borr_mailing_state",
           "borr_mailing_zip", "subject_property_zip", "borr_race_american_indian", "borr_race_asian",
           "borr_race_black", "borr_race_native_hawaiian", "borr_race_white", "borr_race_info_not_provided",
           "borr_race_not_applicable", "co_borr_race_american_indian", "co_borr_race_asian", "co_borr_race_black",
           "co_borr_race_native_hawaiian", "co_borr_race_white", "co_borr_race_info_not_provided",
           "co_borr_race_not_applicable", "subject_property_num_units", "down_payment", "closing_cost_program",
           "subject_property_year_built", "occupancy", "estimated_value", "loan_purpose", "funds_released_date",
           "funds_sent_date","underwriting_appraisal_completed_date",
           "purchase_advice_date", "channel", "interest_rate", "application_date", "loan_officer_name", "ltv",
           "appraised_value", "loan_number", "loan_term", "borr_mexican_indicator",
           "borr_puerto_rican_indicator", "borr_cuban_indicator", "borr_other_hispanic_latino_origin_indicator",
           "borr_asian_indian_indicator", "borr_chinese_indicator", "borr_filipino_indicator",
           "borr_japanese_indicator", "borr_korean_indicator", "borr_vietnamese_indicator",
           "borr_other_asian_race_indicator",
           "borr_native_hawaiian_indicator", "borr_guamanian_or_chamorro_indicator", "borr_samoan_indicator",
           "borr_other_pacific_islander_indicator","borr_female_indicator","borr_male","lien_position",
           "borr_do_not_wish","borr_hispanic_or_latino","borr_not_hispanic_or_latino","borr_sex","co_borr_sex",
           "borr_marital_status","amortization_type","msa","assets_minus_liabilities",
           "top_ratio", "dti", "apr", "co_borr_marital_status", "brokered_funded_date", "branch", "division", "roc",
           "exception_requested_perc",
           "pe_reviewed_by_secondary", "referral_type", "years_months_at_job", "present_state", "loan_folder",
           "approval_date", "docs_signing_date",
           "processing_date", "file_started", "fico", "guid", "last_modified"]


# Rename column names
df.columns = headers


# print(df.dtypes)
print(df.describe())
# print(df[:].head())

# Export as CSV
print(df.shape)
# df.to_csv("enc_data.csv")
df.to_csv("enc_data_ALL_2013_2019.csv")
print("--- %s seconds ---" % (time.time() - start_time))

