import requests
import pandas as pd
from pandas.io.json import json_normalize

pd.set_option('display.max_columns', None)
desired_width = 170
pd.set_option('display.width', desired_width)

keys = pd.read_csv('API_Key_Rep_Loop.csv')

review_list = []
for i in range(0, len(keys)):
    loanofficer = keys['LO'][i]
    key = keys['API Key'][i]
    url = 'http://localfeedbackloop.com/api/retrievecustomer'
    params = {'api_key': key, 'return_flag': 'all_feedback'}
    api_request = requests.get(url=url,params=params)
    json_file = api_request.json()

    if len(json_file['customer_data'][0]) > 0:
        try:
            # Multiple reviews
            data = json_file['customer_data'][0].values()
            data = list(data)[0]
            df = pd.concat([pd.Series(d) for d in data], axis=1).fillna(0).T
            df['rating'] = pd.to_numeric(df['rating'])
            df = df[df['rating'] > 0]
            df['LoanOfficer'] = loanofficer
        except BaseException:
            # One review
            df = json_normalize(json_file['customer_data'])
            df['rating'] = pd.to_numeric(df['rating'])
            df = df[df['rating'] > 0]
            df['LoanOfficer'] = loanofficer
    else:
        # No reviews
        continue
    review_list.append(df)

all_reviews = pd.concat(review_list)
print(all_reviews.head(100))

all_reviews.to_csv('rep_loop.csv', encoding='utf-8-sig')