import pandas as pd
import plotly as py
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio



new='cahrt_new.csv'
old='cahrt_old.csv'
df = pd.read_csv(new, error_bad_lines=False)
df2 = pd.read_csv(old, error_bad_lines=False)

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

def accuracy(df,df2):
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=df.max_depth, y=df.accuracy,
                        mode='lines+markers',
                        name='New'))
    fig.add_trace(go.Scatter(x=df2.max_depth, y=df2.accuracy,
                        mode='lines+markers',
                        name='Previous'))
    fig.update_layout(title='Accuracy')
    pio.write_image(fig, 'chart/accuracy.png', scale=2)
    fig.show()

def precision_c(df,df2):
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=df.max_depth, y=df['precision(companion)'],
                        mode='lines+markers',
                        name='New'))
    fig.add_trace(go.Scatter(x=df2.max_depth, y=df2['precision(companion)'],
                        mode='lines+markers',
                        name='Previous'))
    fig.update_layout(title='Precision(companion)')
    pio.write_image(fig, 'chart/precision(companion).png', scale=2)
    fig.show()

def precision_i(df,df2):
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=df.max_depth, y=df['precision(independent)'],
                        mode='lines+markers',
                        name='New'))
    fig.add_trace(go.Scatter(x=df2.max_depth, y=df2['precision(independent)'],
                        mode='lines+markers',
                        name='Previous'))
    fig.update_layout(title='Precision(independent)')
    pio.write_image(fig, 'chart/precision(independent).png', scale=2)
    fig.show()

def macro(df,df2):
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=df.max_depth, y=df['macro avg'],
                        mode='lines+markers',
                        name='New'))
    fig.add_trace(go.Scatter(x=df2.max_depth, y=df2['macro avg'],
                        mode='lines+markers',
                        name='Previous'))
    fig.update_layout(title='Macro Avg')
    pio.write_image(fig, 'chart/macro.png', scale=2)
    fig.show()

# plotscatter(df)
# plotscatter(df2)
accuracy(df,df2)
precision_i(df,df2)
precision_c(df,df2)
macro(df,df2)



# TrainData=TrainData.set_index('pair')
#     TrainData['State']=1
#     TrainData['State'].loc[(TrainData.index !='dev2_dev5') 
#     & (TrainData.index !='dev2_dev6') 
#     & (TrainData.index !='dev5_dev6') 
#     & (TrainData.index !='dev9_dev10') 
#     & (TrainData.index !='dev9_dev11') 
#     & (TrainData.index !='dev10_dev11')  ]=0