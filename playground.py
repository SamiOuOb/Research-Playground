import pandas as pd
import csv
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio

"""
* Device Sheets *
Name                        Mac                     Code           Term
HTC 10                  |   40:4e:36:08:20:d1   |   HTC 10     |    UE5 (UE2)
Asus Zenfone Max Pro    |   0c:9d:92:72:f9:3a   |   Asus P     |    UE1
Asus Zenfone 2 Laser    |   00:0a:f5:ce:d6:e4   |   Asus L     |    UE4
Asus Padfone S          |   14:dd:a9:bc:11:0c   |   Asus S     |    UE3
Asus Zenfone 6          |   10:c3:7b:2e:e0:00   |   Asus 6     |    UE6
Ipad Mini 1             |   e0:f5:c6:47:94:60   |   Ipad M     |    UE7

"""

start_time = '16:25:00'
end_time = '16:46:00'

wifi = {
    'Jeff'   : '40:4e:36:08:20:d1',
    'Han'    : 'ec:9b:f3:41:c7:2f',
    # 'Sami'   : '0c:9d:92:72:f9:3a',
    'Perry'  : '40:4e:36:94:89:c1',
    'Tony'   : '2c:4d:54:be:e5:3e',
    'Mack'   : '14:dd:a9:bc:11:0c',
    'Jason'  : '30:07:4d:e9:b2:c6',
    'Ray'    : 'd0:17:c2:10:5f:5c'
} 

ble = {
    'Jeff'   : '0000d421-0000-1000-8000-00805f9b34fb',
    'Han'    : '00008e5e-0000-1000-8000-00805f9b34fb',
    'Sami'   : '0000ee76-0000-1000-8000-00805f9b34fb',
    'Perry'  : '0000e4f9-0000-1000-8000-00805f9b34fb',
    'Tony'   : '0000743b-0000-1000-8000-00805f9b34fb',
    # 'Mack'   : '',
    'Jason'  : '000021a1-0000-1000-8000-00805f9b34fb',
    # 'Ray'    : '00009800-0000-1000-8000-00805f9b34fb'
}

ibeacon = {
    'Jeff'   : '12:3b:6a:1a:75:61',
    'Han'    : '12:3b:6a:1a:75:71',
    'Sami1'  : '12:3b:6a:1a:75:66',
    'Sami2'  : '12:3b:6a:1a:75:44',
    'Perry1' : '12:3b:6a:1a:75:5a',
    'Perry2' : '12:3b:6a:1a:75:5b',
    'Tony1'  : '12:3b:6a:1a:62:ed',
    'Tony2'  : '12:3b:6a:1a:62:de',
    'Mack'   : '12:3b:6a:1a:62:ef',
    'Jason'  : '12:3b:6a:1a:63:23',
    'Ray'    : '12:3b:6a:1a:63:1d'
}

device_1_name = 'ibeacon/Han'
device_1_mac = '12:3b:6a:1a:75:71'

device_2_name = 'ibeacon/Perry'
device_2_mac = '12:3b:6a:1a:75:5b'

file0 = "./1127/ble/20181127_100.log"
file1 = "./1127/ble/20181127_101.log"
file2 = "./1127/ble/20181127_102.log"

file3 = "./1127/wifi/20181127_100.log"
file4 = "./1127/wifi/20181127_101.log"
file5 = "./1127/wifi/20181127_102.log"

interval = 5
alpha = 0.5

def getRSSI(filename, ble=False):
    # 整理 log 並上標籤
    df = pd.read_csv(filename, sep='\t', parse_dates=[0], error_bad_lines=False)
    if 'ble' in filename:
        df.columns = ['time','mac','type','RSSI','uuid']
        rate='1'
    else:
        df.columns = ['time','mac','chip','ap','RSSI']
        rate='2'
    df = df.set_index('time')
    df = df.between_time(start_time, end_time)
    # 取出 RSSI 並 resample，取每段時間內最大之RSSI
    if ble == True:
        probe1 = df[df.uuid == device_1_mac].resample(rate+'S').agg(dict(RSSI='max')).dropna()
        probe2 = df[df.uuid == device_2_mac].resample(rate+'S').agg(dict(RSSI='max')).dropna()
    else:
        probe1 = df[df.mac == device_1_mac].resample(rate+'S').agg(dict(RSSI='max')).dropna()
        probe2 = df[df.mac == device_2_mac].resample(rate+'S').agg(dict(RSSI='max')).dropna()
    Orig_P1=probe1.copy()
    Orig_P2=probe2.copy()

    probe1.RSSI = getNormalize(probe1.RSSI)
    probe2.RSSI = getNormalize(probe2.RSSI)
    Nor_P1=probe1
    Nor_P2=probe2

    SES_P1 = getSES(probe1, alpha, False)
    SES_P2 = getSES(probe2, alpha, False)

    return Orig_P1, Orig_P2, Nor_P1, Nor_P2, SES_P1, SES_P2

# RSSI 正規化
def getNormalize(column):
    column = ( column - column.min() ) / ( column.max() - column.min() )
    return column

# 簡易指數平滑，降噪用
def getSES(probe, alpha=None, auto=True):
    # ses=sum([alpha * (1 - alpha) ** i * x for i, x in 
    #             enumerate(reversed(probe.rssi))])
    fit = SimpleExpSmoothing(probe.RSSI).fit(smoothing_level=alpha, optimized=auto)
    ses = fit.fittedvalues.shift(-1)
    ses = pd.DataFrame(ses).reset_index()
    ses.columns = ['time','RSSI']
    ses = ses.set_index('time')
    return ses

# plotly trace
def getShiftScatter(df, name, color):
    trace = go.Scatter(
            y=df['RSSI'],
            x=df.index,
            # marker = dict(color=color),
            # line=dict(shape='spline'),
            mode='lines+markers',
            line=dict(width=0.5),
            marker=dict(size=3),
            name=name,
        )
    return trace

# 主程式
if __name__ == '__main__':

    Orig_P1, Orig_P2, Nor_P1, Nor_P2, SES_P1, SES_P2 = getRSSI(file2)  

    data = [
    # getShiftScatter(df1, 'UE1 - UE5', 'rgb(112,193,71)'),
    # getShiftScatter(df2, 'UE1 - UE6', 'rgb(110,95,169)'),
    # getShiftScatter(df3, 'UE3 - UE6', 'rgb(75,172,198)'),
    getShiftScatter(Orig_P1, 'Han', 'rgb(110,95,169)'),
    getShiftScatter(Orig_P2, 'Perry', 'rgb(75,172,198)'),
    # getShiftScatter(df4, 'UE4 - UE5'),
    # getShiftScatter(df5, 'UE3 - UE5'),
    # getShiftScatter(df6, 'UE4 - UE6', 'rgb(75,172,198)'),
    ]

    data2 = [
    getShiftScatter(Nor_P1, 'Han', 'rgb(110,95,169)'),
    getShiftScatter(Nor_P2, 'Perry', 'rgb(75,172,198)'),
    ]

    data3 = [
    getShiftScatter(SES_P1, 'Han', 'rgb(110,95,169)'),
    getShiftScatter(SES_P2, 'Perry', 'rgb(75,172,198)'),
    ]

    layout = go.Layout(
        xaxis = dict(
            title='Time',
            # range=[0, 5],
            showline = True,
            ),
        yaxis = dict(
            title='RSSI',
            # range=[0.6, 0.8],
            showline = True,
            ),
        font = dict(family='Montserrat SemiBold', size=14),
        # legend=dict(orientation="h"),
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
    )

    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, 'test.png', scale=2)

    fig = go.Figure(data=data2, layout=layout)
    pio.write_image(fig, 'test2.png', scale=2)
    
    fig = go.Figure(data=data3, layout=layout)
    pio.write_image(fig, 'test3.png', scale=2)
    
    

 
    
