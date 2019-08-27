import warnings, os, xlrd, pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now()

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model, preprocessing

import xgboost as xgb

pd.set_option('display.max_columns', None)
desired_width = 170
pd.set_option('display.width', desired_width)

warnings.filterwarnings('ignore')

import time
# Start timer
start_time = time.time()

df = pd.read_csv('enc_data_ALL.csv').drop(['Unnamed: 0'], axis=1)

# Let's take a look at the data of a sample loan
print(df.ix[1])

df['funds_released_date'] = np.where((df['funds_released_date'].notnull() == True), df['funds_released_date'], df['brokered_funded_date'])

# df = df[['file_started','loan_type', 'loan_purpose','amortization_type','ltv','fico','dti','loan_amount','appraised_value','estimated_value','loan_term',
#     'subject_property_state','total_income','borr_dob','apr','funds_released_date','pe_reviewed_by_secondary',
#          'lien_position', 'top_ratio','loan_program','exception_requested_perc',
#          'roc', 'division','msa','assets_minus_liabilities','channel',
#          'borr_marital_status','borr_sex','down_payment', 'application_date','referral_type','loan_officer_name','branch','present_state',
#          'years_months_at_job']]

df = df[["subject_property_type", "loan_amount", "loan_type", "subject_property_city",
         "total_income", "subject_property_state",
         "loan_program", "borr_dob", "borr_race_american_indian", "borr_race_asian",
         "borr_race_black", "borr_race_native_hawaiian", "borr_race_white", "borr_race_info_not_provided",
         "borr_race_not_applicable", "co_borr_race_american_indian", "co_borr_race_asian", "co_borr_race_black",
         "co_borr_race_native_hawaiian", "co_borr_race_white", "co_borr_race_info_not_provided",
         "co_borr_race_not_applicable", "subject_property_num_units", "down_payment",
         "subject_property_year_built", "occupancy", "estimated_value", "loan_purpose", "funds_released_date",
         "channel", "interest_rate", "application_date", "loan_officer_name", "ltv", "appraised_value",
         "loan_term", "borr_mexican_indicator", "borr_puerto_rican_indicator", "borr_cuban_indicator",
         "borr_other_hispanic_latino_origin_indicator", "borr_asian_indian_indicator", "borr_chinese_indicator",
         "borr_filipino_indicator", "borr_japanese_indicator", "borr_korean_indicator", "borr_vietnamese_indicator",
         "borr_other_asian_race_indicator", "borr_native_hawaiian_indicator", "borr_guamanian_or_chamorro_indicator",
         "borr_samoan_indicator", "borr_other_pacific_islander_indicator", "lien_position",
         "borr_do_not_wish","borr_hispanic_or_latino","borr_not_hispanic_or_latino","borr_sex","co_borr_sex",
         "borr_marital_status","amortization_type","msa","assets_minus_liabilities",
         "top_ratio", "dti", "apr", "branch", "division", "roc",
         "referral_type", "years_months_at_job", "present_state", "processing_date", "file_started",
         "fico", "exception_requested_perc", "pe_reviewed_by_secondary"]]


print(df.head(5))

# --------------------------------------------- Preparing The Data ----------------------------------------------------
# Before begining to train models we should transform our data in a way that can be fed into a Machine Learning model.
# The most common techniques are:
# Dealing with missing data and outliers
# It is common to miss some values of data. It may be due to various reasons like errors on the data collection, measurements not applicable, etc. For the same reasons, there can be outliers with extreme values or values that doesn't make sense. For example, a FICO score of 851 is invalid but it will not be recognized by our model.
#
# Missing values are typically represented with the NaN or Null. The problem is that most algorithms can’t handle missing values. Therefore, we need to take care of them before feeding data to our models. Once they are identified, there are several ways to deal with them:
#
# 1) Eliminating the samples or features with missing values/outliers. (we risk to lose relevant information or too many samples) [We are using this method below]
#
# 2) Populating the missing values/outliers with some other values. One common approach is to set them as the mean/median/mode value of the rest of the samples.
# print(df.head())

df = df.replace('na', np.nan)

df["file_started_year"] = pd.to_datetime(df['file_started']).dt.year
df['subject_property_year_built'] = df['subject_property_year_built'].astype(str)
df['subject_property_year_built'] = df['subject_property_year_built'].str.extract('(\d+)',expand=False)
df["subject_property_year_built"] = df["subject_property_year_built"].replace(np.nan, 0.0)
df['subject_property_year_built'] = df['subject_property_year_built'].astype(int)
df['subject_property_year_built'] = np.where((df['subject_property_year_built'] <= 1000), 0.0,
                                             df['subject_property_year_built'])
df['age_of_property'] = np.where((df['subject_property_year_built'] > 0.0), (abs(df['file_started_year']
                                                                    - df['subject_property_year_built'])),0.0)
df = df.drop(['file_started_year'], axis=1)
df = df.drop(['subject_property_year_built'], axis=1)

df['borr_race_american_indian'] = np.where((df['borr_race_american_indian'] == 'Y'), 1.0, 0.0)
df['borr_race_asian'] = np.where((df['borr_race_asian'] == 'Y'), 1.0, 0.0)
df['borr_race_black'] = np.where((df['borr_race_black'] == 'Y'), 1.0, 0.0)
df['borr_race_native_hawaiian'] = np.where((df['borr_race_native_hawaiian'] == 'Y'), 1.0, 0.0)
df['borr_race_white'] = np.where((df['borr_race_white'] == 'Y'), 1.0, 0.0)
df['borr_race_info_not_provided'] = np.where((df['borr_race_info_not_provided'] == 'Y'), 1.0, 0.0)
df['borr_race_not_applicable'] = np.where((df['borr_race_not_applicable'] == 'Y'), 1.0, 0.0)
df['co_borr_race_american_indian'] = np.where((df['co_borr_race_american_indian'] == 'Y'), 1.0, 0.0)
df['co_borr_race_asian'] = np.where((df['co_borr_race_asian'] == 'Y'), 1.0, 0.0)
df['co_borr_race_black'] = np.where((df['co_borr_race_black'] == 'Y'), 1.0, 0.0)
df['co_borr_race_native_hawaiian'] = np.where((df['co_borr_race_native_hawaiian'] == 'Y'), 1.0, 0.0)
df['co_borr_race_white'] = np.where((df['co_borr_race_white'] == 'Y'), 1.0, 0.0)
df['co_borr_race_info_not_provided'] = np.where((df['co_borr_race_info_not_provided'] == 'Y'), 1.0, 0.0)
df['co_borr_race_not_applicable'] = np.where((df['co_borr_race_not_applicable'] == 'Y'), 1.0, 0.0)
df['borr_mexican_indicator'] = np.where((df['borr_mexican_indicator'] == 'Y'), 1.0, 0.0)
df['borr_puerto_rican_indicator'] = np.where((df['borr_puerto_rican_indicator'] == 'Y'), 1.0, 0.0)
df['borr_cuban_indicator'] = np.where((df['borr_cuban_indicator'] == 'Y'), 1.0, 0.0)
df['borr_other_hispanic_latino_origin_indicator'] = np.where((df['borr_other_hispanic_latino_origin_indicator'] == 'Y'), 1.0, 0.0)
df['borr_chinese_indicator'] = np.where((df['borr_chinese_indicator'] == 'Y'), 1.0, 0.0)
df['borr_filipino_indicator'] = np.where((df['borr_filipino_indicator'] == 'Y'), 1.0, 0.0)
df['borr_japanese_indicator'] = np.where((df['borr_japanese_indicator'] == 'Y'), 1.0, 0.0)
df['borr_korean_indicator'] = np.where((df['borr_korean_indicator'] == 'Y'), 1.0, 0.0)
df['borr_vietnamese_indicator'] = np.where((df['borr_vietnamese_indicator'] == 'Y'), 1.0, 0.0)
df['borr_other_asian_race_indicator'] = np.where((df['borr_other_asian_race_indicator'] == 'Y'), 1.0, 0.0)
df['borr_native_hawaiian_indicator'] = np.where((df['borr_native_hawaiian_indicator'] == 'Y'), 1.0, 0.0)
df['borr_guamanian_or_chamorro_indicator'] = np.where((df['borr_guamanian_or_chamorro_indicator'] == 'Y'), 1.0, 0.0)
df['borr_samoan_indicator'] = np.where((df['borr_samoan_indicator'] == 'Y'), 1.0, 0.0)
df['borr_other_pacific_islander_indicator'] = np.where((df['borr_other_pacific_islander_indicator'] == 'Y'), 1.0, 0.0)
df['borr_asian_indian_indicator'] = np.where((df['borr_asian_indian_indicator'] == 'Y'), 1.0, 0.0)
df['borr_do_not_wish'] = np.where((df['borr_do_not_wish'] == 'Y'), 1.0, 0.0)
df['borr_hispanic_or_latino'] = np.where((df['borr_hispanic_or_latino'] == 'Y'), 1.0, 0.0)
df['borr_not_hispanic_or_latino'] = np.where((df['borr_not_hispanic_or_latino'] == 'Y'), 1.0, 0.0)

# df['subject_property_zip'] = df['subject_property_zip'][:5]

df['same_state'] = np.where((df['subject_property_state'] == df['present_state']), 1.0, 0.0)
df = df.drop(['present_state'], axis=1)

df['LoanStatus'] = df['funds_released_date'].notnull().astype("int")
df = df.drop(['funds_released_date'], axis=1)

df["processing_date"] = pd.to_datetime(df["processing_date"]).dt.date
df["file_started"] = pd.to_datetime(df["file_started"]).dt.date
df["today"] = datetime.date.today()
df['days_to_processing'] = np.where((df['processing_date'].notnull() == True), (abs(df['processing_date'] -
                                                                                    df['file_started']).dt.days),
                                    np.where((abs(df['today'] - df['file_started']).dt.days)<= 3,
                                             (abs(df['today'] - df['file_started']).dt.days), '999'))

df = df.drop(['today'], axis=1)
df = df.drop(['processing_date'], axis=1)

df['application_date'] = df['application_date'].notnull().astype("int")

df['LoanStatus'] = np.where((df['LoanStatus'] == 1), 'Funded',
                            (np.where((df['LoanStatus'] == 0), 'Not Funded',df['LoanStatus'])))

df["pe_reviewed_by_secondary"] = np.where((df["pe_reviewed_by_secondary"] == 'Y'), 1.0, 0.0)

# df = df.dropna(how='any').reset_index(drop=True)

df['file_started'] = pd.DatetimeIndex(df['file_started']).month
df = pd.concat([df, pd.get_dummies(df['file_started'])], axis=1)
df.rename(columns={1:'month_jan'}, inplace=True)
df.rename(columns={2:'month_feb'}, inplace=True)
df.rename(columns={3:'month_mar'}, inplace=True)
df.rename(columns={4:'month_apr'}, inplace=True)
df.rename(columns={5:'month_may'}, inplace=True)
df.rename(columns={6:'month_jun'}, inplace=True)
df.rename(columns={7:'month_jul'}, inplace=True)
df.rename(columns={8:'month_aug'}, inplace=True)
df.rename(columns={9:'month_sep'}, inplace=True)
df.rename(columns={10:'month_oct'}, inplace=True)
df.rename(columns={11:'month_nov'}, inplace=True)
df.rename(columns={12:'month_dec'}, inplace=True)
df = df.drop(['file_started'], axis=1)

# Change type to float with 2 decimal places
df["loan_amount"] = round(df["loan_amount"].astype("float"), 2)
df["total_income"] = round(df["total_income"].astype("float"), 2)
# df["interest_rate"] = round(df["interest_rate"].astype("float"), 3)
df["dti"] = round(df["dti"].astype("float"), 2)

# Change type to integer
df["fico"] = df["fico"].replace(np.nan, 0.0)
df["fico"] = df["fico"].astype("int")

# Create Age as Categorical
# df["borr_dob"] = pd.to_datetime(df["borr_dob"]).dt.date
df["borr_dob"] = pd.to_datetime(df['borr_dob']).dt.year
df = df[df.borr_dob <= (now.year - 18)]
df['age_bucket'] = np.where((df['borr_dob'] >= 1946) & (df['borr_dob'] <= 1964), 'Baby Boomers',
                            (np.where((df['borr_dob'] >= 1965) & (df['borr_dob'] <= 1976), 'Generation X',
                                      (np.where((df['borr_dob'] >= 1977) & (df['borr_dob'] <= 1995), 'Millennials',
                                                (np.where((df['borr_dob'] >= 1996) & (df['borr_dob'] <= now.year), 'Generation Z',
                                                          (np.where((df['borr_dob'] <= 1945), 'Traditionalists', df['borr_dob'])))))))))

df = df.drop(['borr_dob'], axis=1)
# Filter out inaccurate information in data
# df = df[df.fico >= 300]
# df = df[df.dti <= 100.0]
# df = df[df.dti > 0.0]

df = df.join(pd.get_dummies(df['loan_type']))
df = df.drop(['loan_type'], axis=1)
df = df.rename(index=str, columns={"Other": "OtherLoanType"})
df = df.rename(index=str, columns={"FarmersHomeAdministration": "USDA"})
df = df.rename(index=str, columns={"VA": "VALoanType"})

# print('Unique values of loan purposes', np.unique(df.loan_purpose))
df['loan_purpose'] = np.where((df['loan_purpose'] == 'Cash-Out Refinance'), 'Refinance',
                            (np.where((df['loan_purpose'] == 'NoCash-Out Refinance'), 'Refinance', df['loan_purpose'])))

df = df.join(pd.get_dummies(df['loan_purpose']))
df = df.drop(['loan_purpose'], axis=1)

df['Reverse'] = np.where((df['loan_program'].str.contains('Reverse')), 1.0, 0.0)
df = df.drop(['loan_program'], axis=1)

df = df.join(pd.get_dummies(df['lien_position']))
df = df.drop(['lien_position'], axis=1)

df = df.join(pd.get_dummies(df['amortization_type']))
df = df.drop(['amortization_type'], axis=1)

df = df.join(pd.get_dummies(df['age_bucket']))
df = df.drop(['age_bucket'], axis=1)

df = df.join(pd.get_dummies(df['subject_property_state']))
df = df.drop(['subject_property_state'], axis=1)
df = df.rename(index=str, columns={"GA": "GAstate"})

df = df.join(pd.get_dummies(df['roc']))
df = df.drop(['roc'], axis=1)

df = df.join(pd.get_dummies(df['division']))
df = df.drop(['division'], axis=1)

df["msa"] = df["msa"].replace(np.nan, 'No MSA')

df['msa'] = df['msa'].astype('str')
df = df.join(pd.get_dummies(df['msa']))
df = df.drop(['msa'], axis=1)

df = df.join(pd.get_dummies(df['channel']))
df = df.drop(['channel'], axis=1)

df = df.join(pd.get_dummies(df['borr_marital_status']))
df = df.drop(['borr_marital_status'], axis=1)

df["borr_sex"] = df["borr_sex"].replace('Female', 'Borr Female')
df["borr_sex"] = df["borr_sex"].replace('Male', 'Borr Male')
df["borr_sex"] = df["borr_sex"].replace('InformationNotProvidedUnknown', 'BorrSex Not Provided')
df["borr_sex"] = df["borr_sex"].replace('No co-applicant', 'BorrSex Not Provided')
df["borr_sex"] = df["borr_sex"].replace('NotApplicable', 'BorrSex Not Provided')
df = df.join(pd.get_dummies(df['borr_sex']))
df = df.drop(['borr_sex'], axis=1)

df["co_borr_sex"] = df["co_borr_sex"].replace('Female', 'CoBorr Female')
df["co_borr_sex"] = df["co_borr_sex"].replace('Male', 'CoBorr Male')
df["co_borr_sex"] = df["co_borr_sex"].replace('InformationNotProvidedUnknown', 'CoBorrSex Not Provided')
df["co_borr_sex"] = df["co_borr_sex"].replace('No co-applicant', 'CoBorrSex Not Provided')
df["co_borr_sex"] = df["co_borr_sex"].replace('NotApplicable', 'CoBorrSex Not Provided')
df = df.join(pd.get_dummies(df['co_borr_sex']))
df = df.drop(['co_borr_sex'], axis=1)

df["referral_type"] = df["referral_type"].replace(np.nan, 'No Referral')
df = df.join(pd.get_dummies(df['referral_type']))
df = df.drop(['referral_type'], axis=1)

df = df.join(pd.get_dummies(df['loan_officer_name']))
df = df.drop(['loan_officer_name'], axis=1)

df["branch"] = df["branch"].replace('KCNORTH', 'KCNORTH Branch')
df["branch"] = df["branch"].replace('LO Not Set', 'No Branch')
df["branch"] = df["branch"].replace('KC3', 'KC3 Branch')
df["branch"] = df["branch"].replace('KC6', 'KC6 Branch')
df["branch"] = df["branch"].replace('MN1', 'MN1 Branch')
df["branch"] = df["branch"].replace('STL1', 'STL1 Branch')
df["branch"] = df["branch"].replace('Train', 'No Branch')
df["branch"] = df["branch"].replace('ERROR', 'No Branch')
df["branch"] = df["branch"].replace('KC1 AW', 'KC1 AW Branch')
df["branch"] = df["branch"].replace('ARLINGTON', 'ARLINGTON Branch')
df = df.join(pd.get_dummies(df['branch']))
df = df.drop(['branch'], axis=1)

df = df.join(pd.get_dummies(df['subject_property_type']))
df = df.drop(['subject_property_type'], axis=1)

df["subject_property_city"] = df["subject_property_city"].replace('DC', np.nan)
df["subject_property_city"] = df["subject_property_city"].replace('TX', np.nan)
df["subject_property_city"] = df["subject_property_city"].replace('AUSTIN', 'Austin City')
df["subject_property_city"] = df["subject_property_city"].replace('DALLAS', 'Dallas City')
df["subject_property_city"] = df["subject_property_city"].replace('FEDERAL WAY', 'Federal Way City')
df["subject_property_city"] = df["subject_property_city"].replace('HOUSTON', 'Houston City')
df["subject_property_city"] = df["subject_property_city"].replace('SPOKANE', 'Spokane City')
df = df.join(pd.get_dummies(df['subject_property_city']))
df = df.drop(['subject_property_city'], axis=1)

# df = df.join(pd.get_dummies(df['subject_property_county_code']))
# df = df.drop(['subject_property_county_code'], axis=1)

df = df.join(pd.get_dummies(df['occupancy']))
df = df.drop(['occupancy'], axis=1)

# df = df.join(pd.get_dummies(df['loan_term']))
# df = df.drop(['loan_term'], axis=1)

df = df.replace(np.nan, 0.0)

model_columns = df.columns
model_columns = pd.DataFrame(model_columns)
model_columns.to_csv("model_columns.csv")

LABEL_COLUMN = 'LoanStatus'

# print(df.ix[0])
# print(df.shape)

# ----------------------------------- Feature Scaling -----------------------------------------------------------
df['dti'] = preprocessing.scale(df['dti'])
df['loan_amount'] = preprocessing.scale(df['loan_amount'])
df['total_income'] = preprocessing.scale(df['total_income'])
df['down_payment'] = preprocessing.scale(df['down_payment'])
df['estimated_value'] = preprocessing.scale(df['estimated_value'])
df['ltv'] = preprocessing.scale(df['ltv'])
df['appraised_value'] = preprocessing.scale(df['appraised_value'])
df['loan_term'] = preprocessing.scale(df['loan_term'])
df['assets_minus_liabilities'] = preprocessing.scale(df['assets_minus_liabilities'])
df['top_ratio'] = preprocessing.scale(df['top_ratio'])
df['apr'] = preprocessing.scale(df['apr'])
df['years_months_at_job'] = preprocessing.scale(df['years_months_at_job'])
df['fico'] = preprocessing.scale(df['fico'])
df['age_of_property'] = preprocessing.scale(df['age_of_property'])
df['days_to_processing'] = preprocessing.scale(df['days_to_processing'])


# from sklearn import preprocessing
# df = preprocessing.StandardScaler().fit(df).transform(df)


funded_loans = df.ix[df['LoanStatus'] == 'Funded',:]
non_funded_loans = df.ix[df['LoanStatus'] == 'Not Funded',:]

# n = 15000
# non_funded_loans = pd.concat([non_funded_loans,non_funded_loans.iloc[np.random.randint(len(non_funded_loans),size=n)]])

# Now combine closed and non_closed loans by concatenating
# the closed/non-closed loans into one dataframe
# df = pd.concat([non_funded_loans, funded_loans])

# Reset index and then shuffle the data
# df = df.copy().reset_index(drop=True)
# df = df.sample(frac=1).reset_index(drop=True)
# df[LABEL_COLUMN] = (df[LABEL_COLUMN].apply(lambda x: x == 'Funded')).astype(int)

from sklearn.utils import shuffle
df = shuffle(df)

# Undersampling majority class for imbalance
count_class_0, count_class_1 = df['LoanStatus'].value_counts()
# df['LoanStatus'].value_counts().plot(kind='bar', title='Count (target)')

df_class_1 = df.ix[df['LoanStatus'] == 'Funded',:]
df_class_0 = df.ix[df['LoanStatus'] == 'Not Funded',:]

df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under['LoanStatus'].value_counts())

# df_test_under['LoanStatus'].value_counts().plot(kind='bar', title='Count (target)')

df = pd.DataFrame(df_test_under)
df['LoanStatus'] = df['LoanStatus'].replace('Funded', 1.0)
df['LoanStatus'] = df['LoanStatus'].replace('Not Funded', 0.0)
#
train, test = train_test_split(df, test_size = 0.2)
train, test = train.reset_index(), test.reset_index()
train, test = train.drop('index', axis=1), test.reset_index().drop(['index', 'level_0'], axis=1)

print('  train shape ={}, test shape={}'.format(train.shape, test.shape))

X = train.ix[:, train.columns != LABEL_COLUMN]
Y = train[LABEL_COLUMN]

# ---------------------------------------------------------------------------------------------------------

# SMOTE

# from imblearn.over_sampling import SMOTE
# os = SMOTE(random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# columns = X_train.columns
# os_data_X,os_data_y=os.fit_sample(X_train, y_train)
# os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
# os_data_y= pd.DataFrame(data=os_data_y,columns=['Y'])
#
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
#
# logreg = LogisticRegression()
#
# rfe = RFE(logreg, 20)
# rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
# print(rfe.support_)
# print(rfe.ranking_)
#
# df_not_closed = df.loc[:, df.columns != 'LoanStatus']
#
# features_bool = np.array(rfe.support_)
# features = np.array(df_not_closed.columns)
#
# print(features_bool.shape)
# print(features.shape)
#
# cols = features[features_bool]
# print(cols)

 # -------------------------------------------------------------------------------------------------------


def modelfit(model, data, predictors, outcome, useTrainCV=True, n_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(data[predictors].values, label=outcome)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=n_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print(cvresult.shape[0] - 1)
        # print (cvresult.iloc(cvresult.shape[0]-1))
        model.set_params(n_estimators=cvresult.shape[0])
        # print (cvresult)

    # Fit the model on data
    model.fit(data[predictors], outcome, eval_metric='auc')

    # Predict training set
    predictions = model.predict(data[predictors])
    probs = model.predict_proba(data[predictors])[:, 1]

    print("\nModel report: ")
    print("Accuracy: {}".format(round(metrics.accuracy_score(outcome, predictions) * 100.0, 2)))
    print("AUC score (train): %f" % metrics.roc_auc_score(outcome, probs))

    my_colors = ['purple', 'magenta', 'yellow', 'orange', 'blue', 'green', 'red']
    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values()
    feat_imp.plot(kind="bar", title="Important features", color=my_colors)
    plt.ylabel("Feature Importance Score")
    return model


import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def print_model_metrics(preds, probs, model, labels, is_xgb=False):
    print("\nModel Report")
    print("Accuracy of the model: {}".format(round(metrics.accuracy_score(labels, preds) * 100.0, 2)))
    print("AUC score: %f" % metrics.roc_auc_score(labels, probs))
    print("Matthews Correlation coefficient: %f" % metrics.matthews_corrcoef(labels, preds))
    print("\nPrecision, Recall, F1beta, Support: ")
    print(metrics.precision_recall_fscore_support(labels, preds, average='weighted'))
    print("F1-score: %f" % metrics.f1_score(labels, preds, average='weighted'))

    if is_xgb:
        my_colors = ['purple', 'magenta', 'yellow', 'orange', 'blue', 'green', 'red']
        feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values()
        feat_imp.plot(kind="bar", title="Important features", color=my_colors)
        plt.ylabel("Feature Importance Score")

    cnf_matrix = metrics.confusion_matrix(labels, preds)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Loan Not Funded', 'Loan Funded'])


#--------------------------------------------------------------------------------------------------------------------
# RANDOM FOREST CLASSIFIER

print('\n\nRANDOM FOREST CLASSIFIER MODEL...')

from sklearn.ensemble import RandomForestClassifier

# df = df.loc[:, df.columns != 'LoanStatus'].columns
predictors = df.loc[:, df.columns != 'LoanStatus'].columns
#  old {'n_estimators': 2000, 'min_samples_split': 10, 'min_samples_leaf': 30, 'max_features': 'auto', 'max_depth': 20, 'bootstrap': True}
#  new with tuning{'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 40, 'bootstrap': True}

rdf = RandomForestClassifier(bootstrap=True,
            class_weight=None,
            criterion='gini',
            max_depth=40,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            min_samples_leaf=4,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=600,
            oob_score=False,
            random_state=0,
            verbose=0,
            warm_start=False)

# X = df.ix[:, df.columns != LABEL_COLUMN]
# Y = df[LABEL_COLUMN]

forest_model = rdf.fit(X[predictors], Y)

print("--- %s seconds ---" % (time.time() - start_time))
#
# test_y = rdf.predict(test[predictors])
# # test_probs = rdf.predict_proba(test[predictors])[:,1]
#
# y_true = test['LoanStatus']
#
# from sklearn import metrics
# print("\nModel Report")
# print("Accuracy of the model: {}".format(round(metrics.accuracy_score(y_true, test_y) * 100.0, 2)))
# print("AUC score: %f" % metrics.roc_auc_score(y_true, test_y))
# print("Matthews Correlation coefficient: %f" % metrics.matthews_corrcoef(y_true, test_y))
# print("\nPrecision, Recall, F1beta, Support: ")
# print(metrics.precision_recall_fscore_support(y_true, test_y, average='weighted'))
# print("F1-score: %f" % metrics.f1_score(y_true, test_y, average='weighted'))
#
# from sklearn.metrics import classification_report
# target_names = ['Not Funded', 'Funded']
# print(classification_report(y_true, test_y, target_names=target_names))
#
# r2 = metrics.r2_score(test['LoanStatus'], test_y)
# print("R squared: ", r2)

# from sklearn.metrics import confusion_matrix
# cnf_matrix = confusion_matrix(test['LoanStatus'], test_y)
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=['Loan Not Funded', 'Loan Funded'])

# PLOT FEATURE IMPORTANCE
feat_importances = pd.Series(rdf.feature_importances_, index=predictors)
# feat_importances.nlargest(50).plot(kind='barh')
# plt.show()
# df_ser = pd.DataFrame(feat_importances.nlargest(50))
# feat_list = df_ser.index.values.tolist()
#
# print("Feature List Top 50: ", feat_list)
#
# # BOTTOM HALF FEATURES
# feat_importances.nsmallest(50).plot(kind='barh')
# plt.show()
# df_ser = pd.DataFrame(feat_importances.nsmallest(4166))
# feat_list2 = df_ser.index.values.tolist()
#
# print("Feature List Bottom 4166 (half): ", feat_list2)


df_ser = pd.DataFrame(feat_importances.nlargest(350))
feat_list = df_ser.index.values.tolist()
df_feat_list = pd.DataFrame(feat_list)
df_feat_list.to_csv("feature_list_400.csv")

predictors = feat_list

model_columns = predictors
model_columns = pd.DataFrame(model_columns)
model_columns.to_csv("model_columns2.csv")

print('\n\nSECOND RANDOM FOREST CLASSIFIER MODEL...')
# old {'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 40, 'bootstrap': False}

rdf = RandomForestClassifier(bootstrap=True,
            class_weight=None,
            criterion='gini',
            max_depth=40,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            min_samples_leaf=4,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=600,
            oob_score=False,
            random_state=0,
            verbose=0,
            warm_start=False)


forest_model2 = rdf.fit(X[predictors], Y)

print("--- %s seconds ---" % (time.time() - start_time))

test_y = rdf.predict(test[predictors])

y_true = test['LoanStatus']

from sklearn import metrics
print("\nModel Report")
print("Accuracy of the model: {}".format(round(metrics.accuracy_score(y_true, test_y) * 100.0, 2)))
print("AUC score: %f" % metrics.roc_auc_score(y_true, test_y))
print("Matthews Correlation coefficient: %f" % metrics.matthews_corrcoef(y_true, test_y))
print("\nPrecision, Recall, F1beta, Support: ")
print(metrics.precision_recall_fscore_support(y_true, test_y, average='weighted'))
print("F1-score: %f" % metrics.f1_score(y_true, test_y, average='weighted'))

from sklearn.metrics import classification_report
target_names = ['Not Funded', 'Funded']
print(classification_report(y_true, test_y, target_names=target_names))

r2 = metrics.r2_score(test['LoanStatus'], test_y)
print("R squared: ", r2)

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(test['LoanStatus'], test_y)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Loan Not Funded', 'Loan Funded'])


#--------------------------------------- SAVE FOREST MODEL TO DISK -------------------------------------------------
import joblib

filename = 'ML_model_random_forest_predict_funded2.dat'
joblib.dump(forest_model2, filename)


#
# -------------------------------------------- RANDOM SEARCH TRAINING --------------------------------------
# from sklearn.model_selection import RandomizedSearchCV
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4, 6, 10]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# print(random_grid)
#
# from sklearn.ensemble import RandomForestRegressor
# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation,
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42)
# # Fit the random search model
# rf_random.fit(X[predictors], Y)
# print(rf_random.best_params_)
# # {'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 40, 'bootstrap': True}

# # -------------------------------------------- Evaluate Random Search -------------------------------------------------
# def evaluate(model, test_features, test_labels):
#     predictions = model.predict(test_features)
#     errors = abs(predictions - test_labels)
#     mape = 100 * np.mean(errors / test_labels)
#     accuracy = 100 - mape
#     print('Model Performance')
#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#     print('Accuracy = {:0.2f}%.'.format(accuracy))
#     return accuracy
#
# base_model = RandomForestRegressor(n_estimators=2000, random_state=0)
# base_model.fit(X[predictors], Y)
# base_accuracy = evaluate(base_model, test[predictors], y_true)
#
# # Model Performance
# # Average Error: 0.1564 degrees.
# # Accuracy = -inf%.
#
# best_random = rf_random.best_estimator_
# random_accuracy = evaluate(best_random, test[predictors], y_true)
#
# # Model Performance
# # Average Error: 0.1558 degrees.
# # Accuracy = -inf%.
#
# print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))
