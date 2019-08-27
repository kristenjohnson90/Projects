import pandas as pd
import time
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt

# Start timer
start_time = time.time()

pd.set_option('display.max_columns', None)
desired_width = 170
pd.set_option('display.width', desired_width)

df = pd.read_csv('final_results_active.csv').drop(['Unnamed: 0'], axis=1)

df["Actual"] = df["Actual"].replace('Funded', 1.0)
df["Actual"] = df["Actual"].replace('Not Funded', 0.0)

actual_fund = df.loc[df['Actual'] == 0.0]
pred_hist = df.hist(column='Probability 1', bins=101, grid=True, figsize=(12,8), color='#003580', zorder=2, rwidth=0.8)
#pred_hist.set_title("Probability to Close of Actual Closings")

hist = pred_hist[0]
for x in hist:
    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)
    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    # Remove title
    x.set_title("Probability to Close")
    # Set x-axis label
    x.set_xlabel("Probability to Close (%)", labelpad=20, size=12)
    # Set y-axis label
    x.set_ylabel("Units", labelpad=20, size=12)
    # Format y-axis label
    # x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

plt.show()


pred_1 = df.loc[df['Prediction'] == 1.0]["Loan Number"].count()
pred_0 = df.loc[df['Prediction'] == 0.0]["Loan Number"].count()
pie_plot = pd.DataFrame({'':[pred_0,pred_1]},index=['Not Closed', 'Closed'])
plot = pie_plot.plot.pie(y='', figsize=(5, 5),colors=['#003580', '#80b5ff'], autopct='%.2f%%', fontsize=12, startangle=90,
                         explode = (0.1,0), shadow=True, title="Current Active Pipeline Prediction \n\n")

df = df.loc[(df['Division'] != '0.0') & (df['Division'] != 'LO Not Set')]
df_close = df.groupby(['Division']).sum()
df_total = df.groupby(['Division']).count()
df_total["Percent Closed"] = (df_close["Prediction"]/df_total["Prediction"])*100.0
df_total.plot.bar(y='Percent Closed', colors=['#003580'],title="Current Active Pipeline Prediction by Division \n\n")
plt.show()


pred_2 = df.loc[df['Prediction'] == 1.0]["loan_amount"].sum()
pred_3 = df.loc[df['Prediction'] == 0.0]["loan_amount"].sum()
vol_total = pred_2 + pred_3
unit_total = df["Loan Number"].count()
pred_fund = df.loc[df['Prediction'] == 1.0]
pred_not_fund = df.loc[df['Prediction'] == 0.0]
volume_pred = pred_fund["loan_amount"].sum()
unit_pred = pred_fund["Loan Number"].count()


print("volume total: ", vol_total)
print("unit total: ", pred_1 + pred_0)
print("predicted to fund: ", pred_1, pred_2)
print("percent of units: ", pred_1/(pred_1 + pred_0))



# ---------------------------------------------------------------------------------------------------------------------

import pandas as pd
import time
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt

# Start timer
start_time = time.time()

pd.set_option('display.max_columns', None)
desired_width = 170
pd.set_option('display.width', desired_width)

df = pd.read_csv('final_results_062019.csv').drop(['Unnamed: 0'], axis=1)

df["Actual"] = df["Actual"].replace('Funded', 1.0)
df["Actual"] = df["Actual"].replace('Not Funded', 0.0)

actual_fund = df.loc[df['Actual'] == 1.0]
pred_hist = actual_fund.hist(column='Probability 1', bins=101, grid=True, figsize=(12,8), color='#003580', zorder=2, rwidth=0.8)
#pred_hist.set_title("Probability to Close of Actual Closings")

hist = pred_hist[0]
for x in hist:
    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)
    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    # Remove title
    x.set_title("Probability to Close of Actual Loans Closed")
    # Set x-axis label
    x.set_xlabel("Probability to Close (%)", labelpad=20, size=12)
    # Set y-axis label
    x.set_ylabel("Units Closed", labelpad=20, size=12)
    # Format y-axis label
    # x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

plt.show()


pred_1 = df.loc[df['Prediction'] == 1.0]["Loan Number"].count()
pred_0 = df.loc[df['Prediction'] == 0.0]["Loan Number"].count()
pie_plot = pd.DataFrame({'':[pred_0,pred_1]},index=['Not Closed', 'Closed'])
plot = pie_plot.plot.pie(y='', figsize=(5, 5),colors=['#003580', '#80b5ff'], autopct='%.2f%%', fontsize=12, startangle=90,
                         explode = (0.1,0), shadow=True, title="Current Active Pipeline Prediction \n\n")

df = df.loc[(df['Division'] != '0') & (df['Division'] != 'LO Not Set')]
df_close = df.groupby(['Division']).sum()
df_total = df.groupby(['Division']).count()
df_total["Percent Closed"] = (df_close["Prediction"]/df_total["Prediction"])*100.0

df_total.plot.bar(y='Percent Closed', colors=['#003580'],title="Current Active Pipeline Prediction by Division \n\n")
plt.show()


vol_total = df["Loan Amount"].sum()
unit_total = df["Loan Number"].count()
pred_fund = df.loc[df['Prediction'] == 1.0]
pred_not_fund = df.loc[df['Prediction'] == 0.0]
volume_pred = pred_fund["Loan Amount"].sum()
unit_pred = pred_fund["Loan Number"].count()


print("volume total: ", vol_total)
print("unit total: ", unit_total)
print("predicted to fund: ", unit_pred, volume_pred)
print("percent of units: ", unit_pred/unit_total)




actual_fund = df.loc[df['Actual'] == 1.0]
plt.scatter(x=df['Probability 1'], y=df['Conventional'])
plt.show()


type_1 = df.loc[df['Prediction'] == 1.0]["Loan Number"].count()
type_0 = df.loc[df['Prediction'] == 0.0]["Loan Number"].count()
df_close = df.groupby(['loan_type']).sum()
df_total = df.groupby(['loan_type']).count()
df_total["Percent Closed"] = (df_close["Prediction"]/df_total["Prediction"])*100.0
df_total.plot.bar(y='Percent Closed', colors=['#003580'],title="Current Active Pipeline Prediction by Mortgage Type \n\n")
plt.show()


df = df.loc[(df['Division'] != '0') & (df['Division'] != 'LO Not Set')]
df_close = df.groupby(['Division']).sum()
df_total = df.groupby(['Division']).count()
df_total["Percent Closed"] = (df_close["Prediction"]/df_total["Prediction"])*100.0
df_total.plot.bar(y='Percent Closed', colors=['#003580'],title="Current Active Pipeline Prediction by Division \n\n")
plt.show()


df_close = df.groupby(['age_bucket']).sum()
df_total = df.groupby(['age_bucket']).count()
df_total["Percent Closed"] = (df_close["Prediction"]/df_total["Prediction"])*100.0
df_total.plot.bar(y='Percent Closed', colors=['#003580'],title="Current Active Pipeline Prediction by Division \n\n")
plt.show()



dfw2 = df.loc[df['Division'] == 'DFW2']
lo_dfw2 = dfw2.loc[dfw2['branch'] == 'DFW2 IN']
df_close = dfw2.groupby(['branch']).sum()
df_total = dfw2.groupby(['branch']).count()
df_total["Percent Closed"] = (df_close["Prediction"]/df_total["Prediction"])*100.0
df_total.plot.bar(y='Percent Closed', colors=['#003580'],title="Current Active Pipeline Prediction by Branch in DFW2 \n\n")
plt.show()
