import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio
from analysis import getRSSI, getMacro, scenario, device_1_name, device_2_name
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 
# py.io.orca.config.executable = 'C:\ProgramData\Miniconda3\orca_app'

def getRawScatter(df, name, mode, group):
    data = go.Scatter(
            x = df.index,
            y = df.rssi,
            name = name,
            mode = mode,
            legendgroup = group,
           )
    return data

def getRawColorScatter(df, name, mode, group, legend, color):
    data = go.Scatter(
            x = df.index,
            y = df.rssi,
            name = name,
            mode = mode,
            legendgroup = group,
            showlegend = legend,
            marker = dict(
                color = color,
                line = dict(color = color),
            )
        )
    return data

def getMeanScatter(df, mode, group, size):
    df.columns = ['dev1','dev2']
    data = [
        go.Scatter(
        x = df.index,
        y = df.dev1,
        name = device_1_name,
        mode = mode,
        legendgroup = group,
        marker = dict(size = size),
        ),
        go.Scatter(
        x = df.index,
        y = df.dev2,
        name = device_2_name,
        mode = mode,
        legendgroup = group,
        marker = dict(size = size),
        ),
    ]
    return data

# dev1_0, dev2_0 = getRSSI("./1119/ble/20181119_100.log")
# dev1_1, dev2_1 = getRSSI("./1119/ble/20181119_101.log")
# dev1_2, dev2_2 = getRSSI("./1119/ble/20181119_102.log")
# dev1_0, dev2_0 = getRSSI("./1119/wifi/20181119_100.log")
# dev1_1, dev2_1 = getRSSI("./1119/wifi/20181119_101.log")
# dev1_2, dev2_2 = getRSSI("./1119/wifi/20181119_102.log")
# file0 = "./1122/ble/20181122_100.log"
# file1 = "./1122/ble/20181122_101.log"
# file2 = "./1122/ble/20181122_102.log"
# dev1_0, dev2_0 = getRSSI("./1122/ble/20181122_100.log")
# dev1_1, dev2_1 = getRSSI("./1122/ble/20181122_101.log")
# dev1_2, dev2_2 = getRSSI("./1122/ble/20181122_102.log")
# print(dev1_2)

file1 = "./1127/ble/20181127_101.log"
dev1_1, dev2_1 = getRSSI("./1127/ble/20181127_101.log")


#* Exponential Smoothing Result
# fit = SimpleExpSmoothing(dev2_0.rssi).fit(smoothing_level=0.5)
# fcast3 = fit.forecast(12)
# ses = fit.fittedvalues.shift(-1)
# ses = pd.DataFrame(ses).reset_index()
# ses.columns = ['time','rssi']
# ses = ses.set_index('time')

# data = [
#     getRawScatter(dev2_0, device_1_name, 'lines+markers','1'),
#     getRawScatter(ses, 'SES', 'lines+markers','2'),
#     # getRawColorScatter(dev1_0, device_1_name, 'lines+markers','1',True,'rgb(203,204,198)'),
#     # getRawColorScatter(ses, 'SES', 'lines+markers','2',True,'rgb(255,188,74)'),
# ]
# layout = go.Layout(
#             # title = '<b>Simple Exponential Smoothing</b>',
#             yaxis = dict(title='RSSI',linecolor='rgb(92,103,115)',range=[0,1.01],gridcolor='rgb(92,103,115)',),
#             xaxis = dict(title='Time',linecolor='rgb(92,103,115)',gridcolor='rgb(92,103,115)'),
#             font = dict(family='Montserrat',color='rgb(203,204,198)'),
#             paper_bgcolor = 'rgba(0,0,0,0)',
#             plot_bgcolor = 'rgba(0,0,0,0)',
#          )
# fig = go.Figure(data=data, layout=layout)
# py.offline.plot(fig, filename='./SimpleExpSmoothing.html')
# pio.write_image(fig, 'ProjectBLE/ses.png', width=800, scale=2)

# #* draw raw scatter chart
# data = [
#     getRawScatter(dev1_1, device_1_name, 'lines+markers','0'),
#     getRawScatter(dev2_1, device_2_name, 'lines+markers','1'),
#     # getRawScatter(dev2_0.shift(10, freq='s'), device_2_name, 'lines+markers','1'),
#     # getRawScatter(dev1_1, device_1_name, 'lines+markers','1'),
#     # getRawScatter(dev2_1, device_2_name, 'lines+markers','1'),
#     # getRawScatter(dev1_2, device_1_name, 'lines+markers','2'),
#     # getRawScatter(dev2_2, device_2_name, 'lines+markers','2'),
# ]
    
# layout = go.Layout(
#             title = '<b>Follower | Similarity = '+str(getMacro())+'</b>',
#             # title = '<b>Scenario | '+scenario+'</b>'+
#             #     '<br>Interval: '+str(interval*2)+
#             #     '       Cosine Similarity: '+str(round(cs_0,3))+', '+str(round(cs_1,3))+', '+str(round(cs_2,3))+
#             #     '       Macro Similarity: '+str(round(MacSim,3))
#             #     ,#+'  Euclidean Distance: '+str(eu),
#             yaxis = dict(title='RSSI'),
#             xaxis = dict(title='Time'),
#             font = dict(family='Montserrat SemiBold', size=12),
#             # legend=dict(orientation="h"),
#          )
# fig = go.Figure(data=data, layout=layout)
# py.offline.plot(fig, filename='./'+scenario+'.html')

# #* mean value Plot
# data = getMeanScatter(mean_0, 'lines+markers', '0', 10)
# layout = go.Layout(
#             title = '<b>Scenario | '+scenario+'</b>',
#             yaxis = dict(title='RSSI(Mean)'),
#             xaxis = dict(title='Time'),
#             font = dict(family='Montserrat'),
#          )
# fig = go.Figure(data=data, layout=layout)
# py.offline.plot(fig, filename='./'+scenario+'mean.html')

# #* raw data vs shift data comparison
# trace1 = getRawColorScatter(dev1_0, device_1_name, 'lines+markers','0',True,'rgb(110,95,169)')
# trace2 = getRawColorScatter(dev2_0, device_2_name, 'lines+markers','1',True,'rgb(80,140,191)')
# trace3 = getRawColorScatter(dev1_0, device_1_name, 'lines+markers','0',False,'rgb(110,95,169)')
# trace4 = getRawColorScatter(dev2_0.shift(21, freq='S'), 'Shifted', 'lines+markers','1',True,'rgb(112,193,71)')
# fig = tools.make_subplots(rows=2, cols=1,shared_xaxes=True)
# fig.append_trace(trace1, 1, 1)
# fig.append_trace(trace2, 1, 1)
# fig.append_trace(trace3, 2, 1)
# fig.append_trace(trace4, 2, 1)
# fig['layout'].update(yaxis = dict(title='RSSI'),
#             xaxis = dict(title='Time'),
#             font = dict(family='Montserrat SemiBold', size=14),
#             # legend=dict(orientation="h"),
#             )
# py.offline.plot(fig, filename='./timeshift.html')

