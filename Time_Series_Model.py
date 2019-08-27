import warnings, os, xlrd, pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now()
import itertools
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
import matplotlib

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model, preprocessing

pd.set_option('display.max_columns', None)
desired_width = 170
pd.set_option('display.width', desired_width)

warnings.filterwarnings('ignore')

import time
# Start timer
start_time = time.time()

df = pd.read_csv('enc_data_ALL_2013_2019.csv').drop(['Unnamed: 0'], axis=1)

df['funds_released_date'] = np.where((df['funds_released_date'].notnull() == True), df['funds_released_date'], df['brokered_funded_date'])

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

df['LoanStatus'] = df['funds_released_date'].notnull().astype("int")

df = df.loc[df['LoanStatus'] == 1]

df = df.replace('na', np.nan)
df = df.replace(np.nan, 0.0)

print(df.head(5))

funded = df.groupby('funds_released_date')['LoanStatus'].sum().reset_index()
funded2 = df.groupby('funds_released_date')['interest_rate'].mean().reset_index()
funded = funded.merge(funded2, left_on='funds_released_date', right_on='funds_released_date')
funded = funded.set_index('funds_released_date')
funded.index = pd.to_datetime(funded.index)


y = funded['LoanStatus'].resample('MS').sum()
y2 = funded['interest_rate'].resample('MS').mean()
y = pd.DataFrame(y).merge(pd.DataFrame(y2), left_on='funds_released_date', right_on='funds_released_date')
y = y['2013-01-01':'2019-07-01']
y['LoanStatus'].plot(figsize=(15, 6))
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y['LoanStatus'], model='additive')
fig = decomposition.plot()
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y['interest_rate'], model='additive')
fig = decomposition.plot()
plt.show()


# --------------------------- ARIMA -----------------------------------------------------------------------------

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

exog = sm.add_constant(y.loc['2013':'2019', 'interest_rate'])
# GRID SEARCH
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(endog = y['LoanStatus'],
                                            exog= exog,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=True,
                                            enforce_invertibility=True)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
                continue

print(results.aic.min())


# ARIMA(0, 1, 1)x(0, 1, 1, 12)12 - AIC:594.5643675867551
# ARIMA(1, 1, 1)x(0, 1, 1, 12)12 - AIC:733.7975707632893

mod = sm.tsa.statespace.SARIMAX(endog = y['LoanStatus'],
                                exog= exog,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
plt.show()


# PREDICTIONS

pred = results.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['LoanStatus'].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Funded Loans')
plt.legend()
plt.show()


y_forecasted = pred.predicted_mean
y_truth = y['LoanStatus']
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# Producing and visualizing forecasts
rates = pd.read_csv('forecasted_rates.csv')#.drop(['Unnamed: 0'], axis=1)
rates = rates.set_index('Date')
rates.index = pd.to_datetime(rates.index)
exog = sm.add_constant(rates['Rate'])

pred_uc = results.get_forecast(steps=29, exog=exog)
pred_ci = pred_uc.conf_int()
ax = y['LoanStatus'].plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Funded Loans')
plt.legend()
plt.show()

pred_ci['predicted'] = pred_ci.mean(axis=1)
print(pred_ci)