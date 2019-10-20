import pandas as pd
import plotly as py
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio



new='cahrt_new.csv'
old='cahrt_old.csv'
add_data='chart_add_data.csv'
chart_new_add_data='chart_new_add_data.csv'
df = pd.read_csv(new, error_bad_lines=False)
df2 = pd.read_csv(old, error_bad_lines=False)
df3 = pd.read_csv(add_data, error_bad_lines=False)
df4 = pd.read_csv(chart_new_add_data, error_bad_lines=False)

def plotscatter(df):
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=df.max_depth, y=df.accuracy,
                        mode='lines+markers',
                        name='accuracy'))
    fig.add_trace(go.Scatter(x=df.max_depth, y=df['macro avg'],
                        mode='lines+markers',
                        name='macro avg'))
    fig.add_trace(go.Scatter(x=df.max_depth, y=df['precision(independent)'],
                        mode='lines+markers',
                        name='precision(independent)'))
    fig.add_trace(go.Scatter(x=df.max_depth, y=df['precision(companion)'],
                        mode='lines+markers',
                        name='precision(companion)'))
    fig.show()

def accuracy(df,df2,df3,df4):
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=df.max_depth, y=df.accuracy,
                        mode='lines+markers',
                        name='New'))
    fig.add_trace(go.Scatter(x=df2.max_depth, y=df2.accuracy,
                        mode='lines+markers',
                        name='Previous'))
    fig.add_trace(go.Scatter(x=df3.max_depth, y=df3.accuracy,
                        mode='lines+markers',
                        name='Add data'))
    fig.add_trace(go.Scatter(x=df4.max_depth, y=df4.accuracy,
                        mode='lines+markers',
                        name='New + Add data'))
    fig.update_layout(title='Accuracy', xaxis = dict(title='Tree Depth'),  yaxis = dict(title='Percentage'), font = dict(family='Montserrat SemiBold',size = 14))
    pio.write_image(fig, 'chart/accuracy.png', scale=2)
    fig.show()

def precision_c(df,df2,df3,df4):
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=df.max_depth, y=df['precision(companion)'],
                        mode='lines+markers',
                        name='New'))
    fig.add_trace(go.Scatter(x=df2.max_depth, y=df2['precision(companion)'],
                        mode='lines+markers',
                        name='Previous'))
    fig.add_trace(go.Scatter(x=df3.max_depth, y=df3['precision(companion)'],
                        mode='lines+markers',
                        name='Add data'))
    fig.add_trace(go.Scatter(x=df4.max_depth, y=df4['precision(companion)'],
                        mode='lines+markers',
                        name='New + Add data'))
    fig.update_layout(title='Precision(companion)', xaxis = dict(title='Tree Depth'),  yaxis = dict(title='Percentage'), font = dict(family='Montserrat SemiBold',size = 14))
    pio.write_image(fig, 'chart/precision(companion).png', scale=2)
    fig.show()

def precision_i(df,df2,df3,df4):
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=df.max_depth, y=df['precision(independent)'],
                        mode='lines+markers',
                        name='New'))
    fig.add_trace(go.Scatter(x=df2.max_depth, y=df2['precision(independent)'],
                        mode='lines+markers',
                        name='Previous'))
    fig.add_trace(go.Scatter(x=df3.max_depth, y=df3['precision(independent)'],
                        mode='lines+markers',
                        name='Add data'))
    fig.add_trace(go.Scatter(x=df4.max_depth, y=df4['precision(independent)'],
                        mode='lines+markers',
                        name='New + Add data'))
    fig.update_layout(title='Precision(independent)', xaxis = dict(title='Tree Depth'),  yaxis = dict(title='Percentage'), font = dict(family='Montserrat SemiBold',size = 14))
    pio.write_image(fig, 'chart/precision(independent).png', scale=2)
    fig.show()

def macro(df,df2,df3,df4):
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=df.max_depth, y=df['macro avg'],
                        mode='lines+markers',
                        name='New'))
    fig.add_trace(go.Scatter(x=df2.max_depth, y=df2['macro avg'],
                        mode='lines+markers',
                        name='Previous'))
    fig.add_trace(go.Scatter(x=df3.max_depth, y=df3['macro avg'],
                        mode='lines+markers',
                        name='Add data'))
    fig.add_trace(go.Scatter(x=df4.max_depth, y=df4['macro avg'],
                        mode='lines+markers',
                        name='New + Add data'))
    fig.update_layout(title='Macro Avg', xaxis = dict(title='Tree Depth'),  yaxis = dict(title='Percentage'), font = dict(family='Montserrat SemiBold',size = 14))
    pio.write_image(fig, 'chart/macro.png', scale=2)
    fig.show()

# plotscatter(df)
# plotscatter(df2)

# accuracy(df,df2)
# precision_i(df,df2)
# precision_c(df,df2)
# macro(df,df2)

accuracy(df,df2,df3,df4)
precision_i(df,df2,df3,df4)
precision_c(df,df2,df3,df4)
macro(df,df2,df3,df4)