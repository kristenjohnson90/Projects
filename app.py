import pandas as pd
import numpy as np
import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# first data pull
exec(open("./EncAPIapp.py").read())
df = pd.read_csv('enc_data_funded.csv').drop(['Unnamed: 0'], axis=1)
df['funds_released_date'] = np.where((df['funds_released_date'].notnull() == True), df['funds_released_date'],
                                      df['brokered_funded_date'])
df = df.groupby(df['division']).count()
df = df.reset_index()

app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Interval(
            id='interval-component',
            interval=300*1000,
            n_intervals=0
        )
])

print(app.layout)


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):
    exec(open("./EncAPIapp.py").read())
    df = pd.read_csv('enc_data_funded.csv').drop(['Unnamed: 0'], axis=1)
    df['funds_released_date'] = np.where((df['funds_released_date'].notnull() == True), df['funds_released_date'],
                                         df['brokered_funded_date'])
    df = df.groupby(df['division']).count()
    df = df.reset_index()

    data = go.Bar(
        x=df['division'],
        y=df['funds_released_date'],
        name='Bar'
        )

    return {
        'data': [data],
        'layout': {
            'title': 'Funded Loans This Month by Division'
        }
    }


if __name__ == '__main__':
    app.run_server(debug=False)
