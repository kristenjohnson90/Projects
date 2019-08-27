import joblib
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now()
import itertools
from sklearn import preprocessing


# Start timer
start_time = time.time()

pd.set_option('display.max_columns', None)
desired_width = 170
pd.set_option('display.width', desired_width)

df = pd.read_csv('enc_data_active_pipeline.csv').drop(['Unnamed: 0'], axis=1)

df['funds_released_date'] = np.where((df['funds_released_date'].notnull() == True), df['funds_released_date'], df['brokered_funded_date'])

df = df[["subject_property_type", "loan_amount", "loan_type", "subject_property_city",
         "total_income", "subject_property_state",
         "loan_program", "borr_dob", "borr_race_american_indian", "borr_race_asian",
         "borr_race_black", "borr_race_native_hawaiian", "borr_race_white", "borr_race_info_not_provided",
         "borr_race_not_applicable", "co_borr_race_american_indian", "co_borr_race_asian", "co_borr_race_black",
         "co_borr_race_native_hawaiian", "co_borr_race_white", "co_borr_race_info_not_provided",
         "co_borr_race_not_applicable", "subject_property_num_units", "down_payment",
         "subject_property_year_built", "occupancy", "estimated_value", "loan_purpose", "funds_released_date",
         "channel", "interest_rate", "application_date", "loan_officer_name", "ltv", "appraised_value", "loan_number",
         "loan_term", "borr_mexican_indicator", "borr_puerto_rican_indicator", "borr_cuban_indicator",
         "borr_other_hispanic_latino_origin_indicator", "borr_asian_indian_indicator", "borr_chinese_indicator",
         "borr_filipino_indicator", "borr_japanese_indicator", "borr_korean_indicator", "borr_vietnamese_indicator",
         "borr_other_asian_race_indicator", "borr_native_hawaiian_indicator", "borr_guamanian_or_chamorro_indicator",
         "borr_samoan_indicator", "borr_other_pacific_islander_indicator", "lien_position",
         "borr_do_not_wish","borr_hispanic_or_latino","borr_not_hispanic_or_latino","borr_sex","co_borr_sex",
         "borr_marital_status","amortization_type","msa","assets_minus_liabilities",
         "top_ratio", "dti", "apr", "branch", "division", "roc",
         "referral_type", "years_months_at_job", "present_state", "processing_date", "file_started",
         "fico", "exception_requested_perc", "pe_reviewed_by_secondary",]]


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
                                    np.where((abs(df['today'] - df['file_started']).dt.days)<= 7,
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

CSV_file = "cleaned_df.csv"
# Save results to CSV
df.to_csv(CSV_file)

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
# df = df.drop(['division'], axis=1)

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

print(df.shape)


model_columns = pd.read_csv('model_columns2.csv').drop(['Unnamed: 0'], axis=1)
model_columns = model_columns.transpose()
# grab the first row for the header
new_header = model_columns.iloc[0]
# take the data less the header row
model_columns = model_columns[1:]
# set the header row as the df header
model_columns.columns = new_header
diff = model_columns.columns.difference(df.columns)

for i in diff:
    df[i] = 0

df_x = df[model_columns.columns]
df_x = df_x.loc[:, df_x.columns != 'LoanStatus']
df_x = df_x.loc[:, df_x.columns != 'loan_number']
df_x = df_x.loc[:, df_x.columns != 'division']
predictors = df_x.columns


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


# ----------------------------------------------------------------------------------------------------


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
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


# ------------------------------------------ load the model from disk pickle file -------------------------------------
file_path = 'ML_model_random_forest_predict_funded2.dat'

# load model from file
loaded_model = joblib.load(file_path)

# make predictions
y_pred = loaded_model.predict(df[predictors])
predictions = [round(value) for value in y_pred]
y_probs = loaded_model.predict_proba(df[predictors])
prob = [(value.astype("float") * 100.0).round(2) for value in y_probs]
prob = pd.DataFrame(prob)

results = []
for i in range(len(predictions)):
    results.append(pd.DataFrame([[df['loan_number'].iloc[i], predictions[i], prob[0].iloc[i], prob[1].iloc[i],
                                  df['LoanStatus'].iloc[i], df['loan_amount'].iloc[i], df['division'].iloc[i]]],
                                columns=['Loan Number', 'Prediction', 'Probability 0', 'Probability 1', 'Actual',
                                         'Loan Amount','Division']))

results = list(results)
results = pd.concat(results, sort=True)

CSV_file = "prediction_results_active.csv"
# Save results to CSV
results.to_csv(CSV_file)



df["LoanStatus"] = df["LoanStatus"].replace('Funded', 1.0)
df["LoanStatus"] = df["LoanStatus"].replace('Not Funded', 0.0)
y_true = df['LoanStatus']

from sklearn import metrics
print("\nModel Report")
print("Accuracy of the model: {}".format(round(metrics.accuracy_score(y_true, y_pred) * 100.0, 2)))
print("AUC score: %f" % metrics.roc_auc_score(y_true, y_pred))
print("Matthews Correlation coefficient: %f" % metrics.matthews_corrcoef(y_true, y_pred))
print("\nPrecision, Recall, F1beta, Support: ")
print(metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted'))
print("F1-score: %f" % metrics.f1_score(y_true, y_pred, average='weighted'))

from sklearn.metrics import classification_report
target_names = ['Not Funded', 'Funded']
print(classification_report(y_true, y_pred, target_names=target_names))

r2 = metrics.r2_score(df['LoanStatus'], y_pred)
print("R squared: ", r2)

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(df['LoanStatus'], y_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Loan Not Funded', 'Loan Funded'])




print("--- %s seconds ---" % (time.time() - start_time))


# MERGE DATA FRAMES -------------------------------------------------------------------------------------------

orig = pd.read_csv('cleaned_df.csv').drop(['Unnamed: 0'], axis=1)
pred = pd.read_csv('prediction_results_active.csv').drop(['Unnamed: 0'], axis=1)

combined = orig.merge(pred, left_on='loan_number', right_on='Loan Number')
combined.to_csv('final_results_active2.csv')

