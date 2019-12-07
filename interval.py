import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.spatial.distance import cosine
from itertools import combinations, permutations
import os
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import pydot

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
start_time2 = '17:00:00'
end_time2 = '17:20:00'

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

mac_queue=['12:3b:6a:1a:75:61', '12:3b:6a:1a:75:71', '12:3b:6a:1a:75:66', '12:3b:6a:1a:75:44',
           '12:3b:6a:1a:75:5a', '12:3b:6a:1a:75:5b', '12:3b:6a:1a:62:ed', '12:3b:6a:1a:62:de',
           '12:3b:6a:1a:62:ef', '12:3b:6a:1a:63:23', '12:3b:6a:1a:63:1d']

mac_queue2=['12:3b:6a:1a:75:71','12:3b:6a:1a:75:5a', '12:3b:6a:1a:75:5b',
            '12:3b:6a:1a:62:ef']

device_1_name = 'ibeacon/Han'
device_1_mac = '12:3b:6a:1a:75:71'

device_2_name = 'ibeacon/Perry2'
device_2_mac = '12:3b:6a:1a:75:5b'

file0 = "./1127/ble/20181127_100.log"
file1 = "./1127/ble/20181127_101.log"
file2 = "./1127/ble/20181127_102.log"

file3 = "./1211/ble/20181211_100.log"
file4 = "./1211/ble/20181211_101.log"
file5 = "./1211/ble/20181211_102.log"

file_queue=[file0,file1,file2]

interval = 5
alpha = 0.5

def getRawData(filename, device_mac):
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

    # 取出兩裝置之 RSSI 並 resample，取每段時間內最大之RSSI
    probe = df[df.mac == device_mac].resample(rate+'S').agg(dict(RSSI='max'))
    probe.RSSI = getNormalize(probe.RSSI).fillna(0)
    SES = getSES(probe.RSSI, alpha, False).dropna()
    # print(SES)
    return SES

def getRawData1211(filename, device_mac):
    # 整理 log 並上標籤
    df = pd.read_csv(filename, sep='\t', parse_dates=[0], error_bad_lines=False)
    if 'ble' in filename:
        df.columns = ['time','mac','type','RSSI','uuid']
        rate='1'
    else:
        df.columns = ['time','mac','chip','ap','RSSI']
        rate='2'
    df = df.set_index('time')
    df = df.between_time(start_time2, end_time2)

    # 取出兩裝置之 RSSI 並 resample，取每段時間內最大之RSSI
    probe = df[df.mac == device_mac].resample(rate+'S').agg(dict(RSSI='max'))
    probe.RSSI = getNormalize(probe.RSSI).fillna(0)
    SES = getSES(probe.RSSI, alpha, False).dropna()
    # print(SES)
    return SES

# RSSI 正規化
def getNormalize(column):
    column = ( column - column.min() ) / ( column.max() - column.min() )
    return column

# 簡易指數平滑，降噪用
def getSES(probe, alpha=None, auto=True):
    # ses=sum([alpha * (1 - alpha) ** i * x for i, x in
    #             enumerate(reversed(probe.rssi))])
    fit = SimpleExpSmoothing(probe).fit(smoothing_level=alpha, optimized=auto)
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

def mergeAllDevData(filename,mac_queue):
    mergedata = getRawData(filename,mac_queue[0])
    for mac in mac_queue[1:] :
        mergedata = pd.merge(mergedata,getRawData(filename,mac),on='time',how='outer').sort_values(by='time').fillna(0)
    mergedata.columns = ['dev1','dev2','dev3','dev4','dev5','dev6','dev7','dev8','dev9','dev10','dev11']
    return mergedata

def mergeAllDevData1211(filename,mac_queue):
    mergedata = getRawData1211(filename,mac_queue[0])
    for mac in mac_queue[1:] :
        mergedata = pd.merge(mergedata,getRawData1211(filename,mac),on='time',how='outer').sort_values(by='time').fillna(0)
    mergedata.columns = ['dev12','dev13','dev14','dev15']
    return mergedata

def getCosSim(SES_P1, SES_P2, interval):
    a_list = SES_P1.tolist()
    b_list = SES_P2.tolist()
    cs = []
    ja=[]
    dev1 = []
    dev2 = []
    for i in range(0, len(a_list), interval):
        cs.append(1 - cosine(a_list[i:i+interval], b_list[i:i+interval]))

        intersection = len(list(set(a_list[i:i+interval]).intersection(b_list[i:i+interval])))
        union = (len(a_list[i:i+interval]) + len(b_list[i:i+interval])) - intersection
        ja.append(float(intersection)/union)

        # for j in range(0, interval):
        dev1.append(a_list[i:i+interval])
        dev2.append(b_list[i:i+interval])
    return cs, ja, dev1, dev2

def getTrainData(filname, mac_queue,interval):
    mergedata=mergeAllDevData(filname,mac_queue)
    dev_name=['dev1','dev2','dev3','dev4','dev5','dev6','dev7','dev8','dev9','dev10','dev11']
    temp = pd.DataFrame()
    TrainData = pd.DataFrame()
    for dev1, dev2 in combinations(dev_name,2) :
        temp['CosSim'], temp['Ja'], temp['dev1'], temp['dev2']=getCosSim(mergedata[dev1],mergedata[dev2],interval)
        temp['pair']='{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2)
        temp['timestamp']=temp.index

        # print(temp)
        # temp=temp.drop(temp.tail(1).index)
        # print(temp)
        TrainData=TrainData.append(temp)
        TrainData=TrainData.drop(temp.tail(1).index)
        
    return TrainData

def getTrainData1211(filname, mac_queue):
    mergedata=mergeAllDevData1211(filname,mac_queue)
    dev_name=['dev12','dev13','dev14','dev15']
    temp = pd.DataFrame()
    TrainData = pd.DataFrame()
    for dev1, dev2 in combinations(dev_name,2) :
        temp['CosSim'], temp['Ja'], temp['dev1'], temp['dev2']=getCosSim(mergedata[dev1],mergedata[dev2])
        temp['pair']='{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2)
        temp['timestamp']=temp.index
        TrainData=TrainData.append(temp)
        TrainData=TrainData.replace(np.PINF, 0)
        TrainData=TrainData.drop(temp.tail(1).index)
    return TrainData

def mergeAllSnifferData(file0,file1,file2,interval):
    RPi100 = getTrainData(file0, mac_queue, interval)
    RPi101 = getTrainData(file1, mac_queue, interval)
    RPi102 = getTrainData(file2, mac_queue, interval)
    sniffer_queue_name=['100', '101', '102']
    sniffer_queue=[RPi100, RPi101 ,RPi102]
    temp = pd.DataFrame()
    TrainData = pd.DataFrame()
    # getTrainData(sniffer_queue,mac_queue)

    for sniffer in sniffer_queue :
        temp=sniffer
        temp['RPi_No']=sniffer_queue_name.pop(0)
        TrainData=TrainData.append(temp)
    
    TrainData=TrainData.set_index('pair')
    TrainData['State']=1
    TrainData['State'].loc[(TrainData.index !='dev2_dev5') 
    & (TrainData.index !='dev2_dev6') 
    & (TrainData.index !='dev5_dev6') 
    & (TrainData.index !='dev9_dev10') 
    & (TrainData.index !='dev9_dev11') 
    & (TrainData.index !='dev10_dev11')  ]=0

    return TrainData

def mergeAllSnifferData1211():
    RPi100 = getTrainData1211(file3, mac_queue2)
    RPi101 = getTrainData1211(file4, mac_queue2)
    RPi102 = getTrainData1211(file5, mac_queue2)
    sniffer_queue_name=['100', '101', '102']
    sniffer_queue=[RPi100, RPi101 ,RPi102]
    temp = pd.DataFrame()
    TrainData = pd.DataFrame()
    # getTrainData(sniffer_queue,mac_queue)

    for sniffer in sniffer_queue :
        temp=sniffer
        temp['RPi_No']=sniffer_queue_name.pop(0)
        TrainData=TrainData.append(temp)
    
    TrainData=TrainData.set_index('pair')
    TrainData['State']=1

    return TrainData

# 定義準確度
def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return(truth == pred).mean()*100
    
    else:
        return 0

def fit_model_k_fold(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    k_fold = KFold(n_splits=10)
    
    #  Create a decision tree clf object
    dtc = DTC(random_state=80)

    # params = {'max_depth':range(10,21),'criterion':np.array(['entropy','gini'])}
    params = {'max_depth':range(5,16),'criterion':np.array(['entropy','gini'])}

    # Transform 'accuracy_score' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(accuracy_score)

    # Create the grid search object
    grid = GridSearchCV(dtc, param_grid=params,scoring=scoring_fnc,cv=k_fold)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# 主程式
if __name__ == '__main__':


    



    # TrainData=mergeAllSnifferData(file0,file1,file2,interval)
    # TrainData.CosSim=TrainData.CosSim.fillna(0)
    # TrainData.to_csv('TrainData.csv')

    # # ===========1211================
    # TrainData1211=mergeAllSnifferData1211()
    # TrainData=mergeAllSnifferData().append(mergeAllSnifferData1211())
    # TrainData.CosSim=TrainData.CosSim.fillna(0)
    # TrainData.to_csv('TrainData1211.csv')
    # print(TrainData1211)

    # print(TrainData)
    # print(TrainData.isnull().any())

    # TrainData = pd.read_csv('TrainData.csv', error_bad_lines=False)
    # TrainData12 = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
    
    # print(TrainData)

    # X = TrainData.iloc[:, 0:13]
    # y = TrainData.iloc[:, 13]
    # # print(X)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # print("Training set has {} samples.".format(X_train.shape[0]))
    # print("Testing set has {} samples.".format(X_test.shape[0]))

    # # # 預設參數之決策樹
    # # dtc = DTC()

    # dtc = fit_model_k_fold(X_train, y_train)
    # dtc.fit(X_train, y_train)
    # print("k_fold Parameter 'max_depth' is {} for the optimal model.".format(dtc.get_params()['max_depth']))
    # print("k_fold Parameter 'criterion' is {} for the optimal model.".format(dtc.get_params()['criterion']))
    # print(dtc)

    # # # 特徵重要度 
    # print(dtc.feature_importances_)
    # predict_target = dtc.predict(X_test)
    # print(predict_target)
    # print(sum(predict_target == y_test))

    # from sklearn import metrics
    # print(metrics.classification_report(y_test, predict_target))
    # print(metrics.confusion_matrix(y_test, predict_target))


    # print('Accuracy：', dtc.score(X_test, y_test))
    # with open('tree.dot', 'w') as f:
    #     f = export_graphviz(dtc, feature_names=X.columns ,out_file=f, filled=True, rounded=True, special_characters=True)
    # # # dot -Tpng tree.dot -o tree.png
    # # # dot -Tsvg tree.dot -o tree.svg
    
    
    # ========== Barcahrt ==========
    # PairData = pd.DataFrame()
    # pair_queue=['dev1','dev2','dev5','dev6']
    # for dev1, dev2 in combinations(pair_queue,2) :
    #     PairData['{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2)] = TrainData[TrainData['pair']=='{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2)].CosSim.tolist()

    # # print(PairData)
    # similarity=[]
    # for pair in PairData.columns:
    #     similarity.append(PairData['{}'.format(pair)].mean())
    # print(similarity)

    # Data = pd.DataFrame()
    # Data['pair']=PairData.columns
    # Data['similarity']=similarity
    # Data['State']=1
    # Data['State'].loc[(Data.pair !='dev2_dev5') 
    # & (Data.pair !='dev2_dev6') 
    # & (Data.pair !='dev3_dev4')
    # & (Data.pair !='dev5_dev6') 
    # & (Data.pair !='dev7_dev8')
    # & (Data.pair !='dev9_dev10') 
    # & (Data.pair !='dev9_dev11') 
    # & (Data.pair !='dev10_dev11')  ]=0

    # Data1=Data[Data.State==0]
    # Data2=Data[Data.State==1]

    # print(Data1)
    # print(Data2)

    # fig = go.Figure()
    # # Add traces
    # fig.add_trace(go.Bar(x=Data1.pair, y=Data1.similarity,
    #                     name='Independent'))
    # fig.add_trace(go.Bar(x=Data2.pair, y=Data2.similarity,
    #                     name='Companion'))
    # fig.update_layout(title='Similarity')
    # pio.write_image(fig, 'chart/Accuracy_WiTrack.png', scale=2)
    # fig.show()


    # # ========== WiTrack validation ==========
    # TrainData = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
    # Data = pd.DataFrame()
    # Data1211 = pd.DataFrame()

    # # pair_queue=['dev1','dev2','dev3','dev4','dev5','dev6','dev7','dev8','dev9','dev10','dev11']
    # pair_queue=['dev1','dev2','dev5','dev6']
    # for dev1, dev2 in combinations(pair_queue,2) :
    #     pair='{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2)
    #     Data=Data.append(TrainData[TrainData['pair'].isin([pair])])
    # # pair_queue=['dev12','dev13','dev14','dev15']
    # # for dev1, dev2 in combinations(pair_queue,2) :
    # #     pair='{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2)
    # #     Data=Data.append(TrainData1211[TrainData1211['pair'].isin([pair])])
    # Data=Data.reset_index()
    # # print(Data)

    # thre_all=[]
    # acu_all=[]
    # for i in range(65,90):
    #     thre=i/100
    #     c=len(Data[(Data.State==1)&(Data.CosSim >= thre)])+len(Data[(Data.State==0)&(Data.CosSim < thre)])
    #     # print(thre)
    #     # print(len(Data[(Data.State==1)&(Data.CosSim >= thre)]))
    #     # print(len(Data[(Data.State==0)&(Data.CosSim < thre)]))
    #     thre_all.append(thre)
    #     # print(c/len(Data))
    #     acu_all.append(c/len(Data))
    # Data_Bar = pd.DataFrame()
    # Data_Bar['Threshold']=thre_all
    # Data_Bar['Accuracy']=acu_all

    # import plotly.express as px
    # fig = px.line(Data_Bar, x='Threshold', y='Accuracy')
    # fig.update_layout(title='WiTrack’s Performance')
    # pio.write_image(fig, 'chart/ThrevsAccu.png', scale=2)
    # fig.show()

    # # ========== Find the most suitable size of training set ==========
    # # import time
    

    # for interval in range(5,30):
        
    #     # tStart = time.time()#計時開始

    #     # TrainData=mergeAllSnifferData(file0,file1,file2,interval)
    #     # TrainData.CosSim=TrainData.CosSim.fillna(0)
    #     # TrainData=TrainData.reset_index()
    #     # for i in TrainData.index:
    #     #     for j in range(1,interval+1):
    #     #         TrainData['dev1_{}'.format(j)] = TrainData.dev1[i].pop(0)
    #     #     for j in range(1,interval+1):
    #     #         TrainData['dev2_{}'.format(j)] = TrainData.dev2[i].pop(0)
    #     # TrainData=TrainData.drop(['dev1', 'dev2'], axis=1)
    #     # TrainData.to_csv('interval_csv/TrainData{}.csv'.format(interval))

    #     # tEnd = time.time()#計時結束
    #     # print("It cost %f sec" % (tEnd - tStart))#會自動做進位

    #     TrainData = pd.read_csv('interval_csv/TrainData{}.csv'.format(interval), error_bad_lines=False)
    #     TrainData=TrainData.set_index('pair')
    #     y = TrainData.State
    #     X = TrainData.drop(['dev1','dev2','State'], axis=1)
    #     X = X.drop('Unnamed: 0', axis=1)
        

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    #     # print("Training set has {} samples.".format(X_train.shape[0]))
    #     # print("Testing set has {} samples.".format(X_test.shape[0]))

    #     dtc = fit_model_k_fold(X_train, y_train)
    #     dtc.fit(X_train, y_train)
    #     print("k_fold Parameter 'max_depth' is {} for the optimal model.".format(dtc.get_params()['max_depth']))
    #     print("k_fold Parameter 'criterion' is {} for the optimal model.".format(dtc.get_params()['criterion']))
    #     # print(dtc)

    #     # # # 特徵重要度
    #     # # print(dtc.feature_importances_)
    #     # # predict_target = dtc.predict(X_test)
    #     # # print(predict_target)
    #     # # print(sum(predict_target == y_test))

    #     # # from sklearn import metrics
    #     # # print(metrics.classification_report(y_test, predict_target))
    #     # # print(metrics.confusion_matrix(y_test, predict_target))

    #     print('If interval={} Accuracy：'.format(interval), dtc.score(X_test, y_test))

    # # ========== TranData Size Comparison ==========
    # xaxis=[]
    # yaxis=[]
    # for timelength in range(60,1260,60):
    #     # timelength=600 #資料長度(秒數)
    #     TrandataSize=timelength/interval
    #     TrainData = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
    #     TrainData=TrainData.set_index('pair')
    #     # TrainData1211 = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
    #     # print(TrainData1211)

    #     TrainData=TrainData.loc[(TrainData.timestamp <= TrandataSize)]
    #     # print(TrainData)

    #     X = TrainData.iloc[:, 0:13]
    #     y = TrainData.iloc[:, 13]
    #     # print(X)

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    #     # print("Training set has {} samples.".format(X_train.shape[0]))
    #     # print("Testing set has {} samples.".format(X_test.shape[0]))

    #     # # # 預設參數之決策樹
    #     # dtc = DTC()

    #     dtc = fit_model_k_fold(X_train, y_train)
    #     dtc.fit(X_train, y_train)
    #     # print("k_fold Parameter 'max_depth' is {} for the optimal model.".format(dtc.get_params()['max_depth']))
    #     # print("k_fold Parameter 'criterion' is {} for the optimal model.".format(dtc.get_params()['criterion']))
    #     # print(dtc)

    #     # # 特徵重要度 
    #     # print(dtc.feature_importances_)
    #     predict_target = dtc.predict(X_test)
    #     # print(predict_target)
    #     # print(sum(predict_target == y_test))

    #     # from sklearn import metrics
    #     # print(metrics.classification_report(y_test, predict_target))
    #     # print(metrics.confusion_matrix(y_test, predict_target))

    #     xaxis.append(timelength/60)
    #     yaxis.append(dtc.score(X_test, y_test))
    #     # print(timelength/60, ' min Accuracy：', dtc.score(X_test, y_test))

    #     # with open('tree.dot', 'w') as f:
    #     #     f = export_graphviz(dtc, feature_names=X.columns ,out_file=f, filled=True, rounded=True, special_characters=True)
    #     # # # dot -Tpng tree.dot -o tree.png
    #     # # # dot -Tsvg tree.dot -o tree.svg

    # import plotly.graph_objects as go
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=xaxis, y=yaxis,
    #                 mode='lines+markers',
    #                 name='With CosSimularity'))
    # fig.update_layout(title='TranData Size Comparison', xaxis = dict(title='Traindata Size (minutes)'),  yaxis = dict(title='Accuracy'), font = dict(family='Montserrat SemiBold',size = 14))
    # pio.write_image(fig, 'chart/TranData Size Comparison.png', scale=2)
    # fig.show()

# # ========== TranData Size Comparison(Without CosSimilarity) ==========
#     xaxis_without=[]
#     yaxis_without=[]
#     for timelength in range(60,1260,60):
#         # timelength=600 #資料長度(秒數)
#         TrandataSize=timelength/interval
#         TrainData = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
#         TrainData=TrainData.set_index('pair')
#         # TrainData1211 = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
#         # print(TrainData1211)

#         TrainData=TrainData.loc[(TrainData.timestamp <= TrandataSize)]
#         # print(TrainData)

#         X = TrainData.iloc[:, 0:13]
#         X=X.drop(['CosSim'], axis=1)
#         y = TrainData.iloc[:, 13]
#         # print(X)

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#         # print("Training set has {} samples.".format(X_train.shape[0]))
#         # print("Testing set has {} samples.".format(X_test.shape[0]))

#         # # # 預設參數之決策樹
#         # dtc = DTC()

#         dtc = fit_model_k_fold(X_train, y_train)
#         dtc.fit(X_train, y_train)
#         print("k_fold Parameter 'max_depth' is {} for the optimal model.".format(dtc.get_params()['max_depth']))
#         # print("k_fold Parameter 'criterion' is {} for the optimal model.".format(dtc.get_params()['criterion']))
#         # print(dtc)

#         # # 特徵重要度
#         # print(dtc.feature_importances_)
#         predict_target = dtc.predict(X_test)
#         # print(predict_target)
#         # print(sum(predict_target == y_test))

#         # from sklearn import metrics
#         # print(metrics.classification_report(y_test, predict_target))
#         # print(metrics.confusion_matrix(y_test, predict_target))

#         xaxis_without.append(timelength/60)
#         yaxis_without.append(dtc.score(X_test, y_test))
#         # print(timelength/60, ' min Accuracy：', dtc.score(X_test, y_test))

#         # with open('tree.dot', 'w') as f:
#         #     f = export_graphviz(dtc, feature_names=X.columns ,out_file=f, filled=True, rounded=True, special_characters=True)
#         # # # dot -Tpng tree.dot -o tree.png
#         # # # dot -Tsvg tree.dot -o tree.svg

#     # import plotly.graph_objects as go
#     # fig = go.Figure()
#     # fig.add_trace(go.Scatter(x=xaxis_without, y=yaxis_without,
#     #                 mode='lines+markers',
#     #                 name='Without CosSimularity'))
#     # fig.update_layout(title='TranData Size Comparison', xaxis = dict(title='Traindata Size (minutes)'),  yaxis = dict(title='Accuracy'), font = dict(family='Montserrat SemiBold',size = 14))
#     # pio.write_image(fig, 'chart/TranData Size Comparison(Without CosSimilarity).png', scale=2)
#     # fig.show()

# # ========== TranData Size Comparison(WiTrack) ==========
#     xaxis=[]
#     acu_all=[]
#     for timelength in range(60,1260,60):
#         # timelength=600 #資料長度(秒數)
#         TrandataSize=timelength/interval
#         TrainData = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
#         # TrainData=TrainData.set_index('pair')
#         TrainData=TrainData.loc[(TrainData.timestamp <= TrandataSize)]

#         Data = pd.DataFrame()
#         Data1211 = pd.DataFrame()

#         # pair_queue=['dev1','dev2','dev3','dev4','dev5','dev6','dev7','dev8','dev9','dev10','dev11']
#         pair_queue=['dev1','dev2','dev5','dev6']
#         for dev1, dev2 in combinations(pair_queue,2) :
#             pair='{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2)
#             Data=Data.append(TrainData[TrainData['pair'].isin([pair])])

#         # pair_queue=['dev12','dev13','dev14','dev15']
#         # for dev1, dev2 in combinations(pair_queue,2) :
#         #     pair='{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2)
#         #     Data=Data.append(TrainData1211[TrainData1211['pair'].isin([pair])])
#         Data=Data.reset_index()

#         thre=80/100
#         c=len(Data[(Data.State==1)&(Data.CosSim >= thre)])+len(Data[(Data.State==0)&(Data.CosSim < thre)])
#         # print(thre)
#         print(len(Data[(Data.State==1)&(Data.CosSim >= thre)]))
#         print(len(Data[(Data.State==0)&(Data.CosSim < thre)]))
#         acu=c/len(Data)

#         xaxis.append(timelength/60)
#         acu_all.append(acu)


#     import plotly.graph_objects as go
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=xaxis, y=acu_all,
#                     mode='lines+markers',
#                     name='WiTrack'))
#     fig.add_trace(go.Scatter(x=xaxis, y=yaxis,
#                     mode='lines+markers',
#                     name='With CosSimularity'))
#     fig.add_trace(go.Scatter(x=xaxis_without, y=yaxis_without,
#                     mode='lines+markers',
#                     name='Without CosSimularity'))
#     fig.update_layout(title='Trandata Size Comparison', xaxis = dict(title='Traindata Size (minutes)'),  yaxis = dict(title='Accuracy'), font = dict(family='Montserrat SemiBold',size = 14))
#     pio.write_image(fig, 'chart/TranData Size Comparison(All).png', scale=2)
#     fig.show()

# ========== TranData Size Comparison ==========
    xaxis=[]
    yaxis=[]

    for timelength in range(60,900,60):
        # timelength=600 #資料長度(秒數)
        TrandataSize=timelength/interval
        TrainData = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
        TrainData=TrainData.set_index('pair')

        X = TrainData.iloc[:, 0:13]
        y = TrainData.iloc[:, 13]
        # print(X)

        X_test=X.loc[(TrainData.timestamp > (900/interval))]
        y_test=y.loc[(TrainData.timestamp > (900/interval))]

        acu_queue=[]
        for i in range(0,16-int(timelength/60)):
            X_train=X.loc[(TrainData.timestamp > i*60/interval) & (TrainData.timestamp <= TrandataSize+i*60/interval)]
            y_train=y.loc[(TrainData.timestamp > i*60/interval) & (TrainData.timestamp <= TrandataSize+i*60/interval)]

            # print("Training set has {} samples.".format(X_train.shape[0]))
            # print("Testing set has {} samples.".format(X_test.shape[0]))

            # # # # 預設參數之決策樹
            # # dtc = DTC()

            dtc = fit_model_k_fold(X_train, y_train)
            dtc.fit(X_train, y_train)
            # print("max_depth is {} for the optimal model.".format(dtc.get_params()['max_depth']))
            # print("k_fold Parameter 'criterion' is {} for the optimal model.".format(dtc.get_params()['criterion']))
            # print(dtc)

            # # 特徵重要度 
            # print(dtc.feature_importances_)
            # predict_target = dtc.predict(X_test)
            # print(predict_target)
            # print(sum(predict_target == y_test))

            # from sklearn import metrics
            # print(metrics.classification_report(y_test, predict_target))
            # print(metrics.confusion_matrix(y_test, predict_target))

            acu_queue.append(dtc.score(X_test, y_test))
            print(timelength/60, ' min Accuracy：', dtc.score(X_test, y_test))

        xaxis.append(timelength/60)
        yaxis.append(sum(acu_queue)/(16-int(timelength/60)))

        # # =================== 將各個裝置pair分開計算各自的預測結果 ======================
        # i=0
        # while((max(TrainData.timestamp)-(TrandataSize+12*i))>0):
        #     X_testset.append(X.loc[(TrainData.timestamp > TrandataSize+12*i) & (TrainData.timestamp <= TrandataSize+12*(i+1))])
        #     y_testset.append(y.loc[(TrainData.timestamp > TrandataSize+12*i) & (TrainData.timestamp <= TrandataSize+12*(i+1))])
        #     i=i+1
        # import collections, numpy
        # accuracy_que=[]
        # for i in range(len(X_testset)):
        #     accuracy_que.append(dtc.score(X_testset[i], y_testset[i]))

        #     pair_queue=['dev1','dev2','dev3','dev4','dev5','dev6','dev7','dev8','dev9','dev10','dev11']
        #     pair_queue2=['dev12','dev13','dev14','dev15']
        #     for dev1, dev2 in combinations(pair_queue,2) :
        #         # X_testset[i].loc[(X_testset[i].index =='{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2))]
        #         test_pair=X_testset[i].loc[(X_testset[i].index =='{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2))]
        #         predict_target = dtc.predict(test_pair)
        #         print('{dev1}_{dev2} = '.format(dev1=dev1,dev2=dev2), '0: ', np.count_nonzero(predict_target==0), '1:', np.count_nonzero(predict_target==1))

        # print(accuracy_que)

        # # with open('tree.dot', 'w') as f:
        # #     f = export_graphviz(dtc, feature_names=X.columns ,out_file=f, filled=True, rounded=True, special_characters=True)
        # # # # dot -Tpng tree.dot -o tree.png
        # # # # dot -Tsvg tree.dot -o tree.svg

    # import plotly.graph_objects as go
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=xaxis, y=yaxis,
    #                 mode='lines+markers',
    #                 name='With CosSimularity'))
    # fig.update_layout(title='TranData Size Comparison', xaxis = dict(title='Traindata Size (minutes)'),  yaxis = dict(title='Accuracy'), font = dict(family='Montserrat SemiBold',size = 14))
    # pio.write_image(fig, 'chart/TranData Size Comparison_v3.png', scale=2)
    # fig.show()

# ========== TranData Size Comparison(Without CosSimilarity) ==========
    xaxis_without=[]
    yaxis_without=[]
    for timelength in range(60,900,60):
        # timelength=600 #資料長度(秒數)
        TrandataSize=timelength/interval
        TrainData = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
        TrainData=TrainData.set_index('pair')
        # TrainData1211 = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
        # print(TrainData1211)

        # print(TrainData)

        X = TrainData.iloc[:, 0:13]
        X=X.drop(['CosSim'], axis=1)
        y = TrainData.iloc[:, 13]
        # print(X)

        X_test=X.loc[(TrainData.timestamp > (900/interval))]
        y_test=y.loc[(TrainData.timestamp > (900/interval))]

        acu_queue=[]
        for i in range(0,16-int(timelength/60)):
            X_train=X.loc[(TrainData.timestamp > i*60/interval) & (TrainData.timestamp <= TrandataSize+i*60/interval)]
            y_train=y.loc[(TrainData.timestamp > i*60/interval) & (TrainData.timestamp <= TrandataSize+i*60/interval)]

            # print("Training set has {} samples.".format(X_train.shape[0]))
            # print("Testing set has {} samples.".format(X_test.shape[0]))

            # # # # 預設參數之決策樹
            # # dtc = DTC()

            dtc = fit_model_k_fold(X_train, y_train)
            dtc.fit(X_train, y_train)
            # print("max_depth is {} for the optimal model.".format(dtc.get_params()['max_depth']))
            # print("k_fold Parameter 'criterion' is {} for the optimal model.".format(dtc.get_params()['criterion']))
            # print(dtc)

            # # 特徵重要度 
            # print(dtc.feature_importances_)
            # predict_target = dtc.predict(X_test)
            # print(predict_target)
            # print(sum(predict_target == y_test))

            # from sklearn import metrics
            # print(metrics.classification_report(y_test, predict_target))
            # print(metrics.confusion_matrix(y_test, predict_target))

            acu_queue.append(dtc.score(X_test, y_test))
            print(timelength/60, ' min Accuracy：', dtc.score(X_test, y_test))

        xaxis_without.append(timelength/60)
        yaxis_without.append(sum(acu_queue)/(16-int(timelength/60)))

#         # with open('tree.dot', 'w') as f:
#         #     f = export_graphviz(dtc, feature_names=X.columns ,out_file=f, filled=True, rounded=True, special_characters=True)
#         # # # dot -Tpng tree.dot -o tree.png
#         # # # dot -Tsvg tree.dot -o tree.svg

    # import plotly.graph_objects as go
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=xaxis_without, y=yaxis_without,
    #                 mode='lines+markers',
    #                 name='Without CosSimularity'))
    # fig.update_layout(title='TranData Size Comparison', xaxis = dict(title='Traindata Size (minutes)'),  yaxis = dict(title='Accuracy'), font = dict(family='Montserrat SemiBold',size = 14))
    # pio.write_image(fig, 'chart/TranData Size Comparison(Without CosSimilarity)_v3.png', scale=2)
    # fig.show()

# ========== TranData Size Comparison(WiTrack) ==========
    xaxis=[]
    acu_all=[]
    for timelength in range(60,900,60):
        # timelength=600 #資料長度(秒數)
        TrandataSize=timelength/interval
        TrainData = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
        # TrainData=TrainData.set_index('pair')

        TestData=TrainData.loc[(TrainData.timestamp > (900/interval))]

        TrainData_temp = pd.DataFrame()
        acu_queue=[]
        for i in range(0,16-int(timelength/60)):
            TrainData_temp=TrainData.loc[(TrainData.timestamp > i*60/interval) & (TrainData.timestamp <= TrandataSize+i*60/interval)]
            # print(TrainData_temp)
            Data = pd.DataFrame()
            test = pd.DataFrame()

            # pair_queue=['dev1','dev2','dev3','dev4','dev5','dev6','dev7','dev8','dev9','dev10','dev11']
            pair_queue=['dev1','dev2','dev5','dev6']
            for dev1, dev2 in combinations(pair_queue,2) :
                pair='{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2)
                Data=Data.append(TrainData_temp[TrainData_temp['pair'].isin([pair])])
                test=test.append(TestData[TestData['pair'].isin([pair])])

            # pair_queue=['dev12','dev13','dev14','dev15']
            # for dev1, dev2 in combinations(pair_queue,2) :
            #     pair='{dev1}_{dev2}'.format(dev1=dev1,dev2=dev2)
            #     Data=Data.append(TrainData1211[TrainData1211['pair'].isin([pair])])
            Data=Data.reset_index()
            test=test.reset_index()

            c=0
            thre=0
            for thre_temp in range(50,95,1):
                thre_temp=thre_temp/100
                c_temp=len(Data[(Data.State==1)&(Data.CosSim >= thre_temp)])+len(Data[(Data.State==0)&(Data.CosSim < thre_temp)])
                if c < c_temp :
                    c=c_temp
                    thre=thre_temp
            print('thre=',thre)

            c=len(test[(test.State==1)&(test.CosSim >= thre)])+len(test[(test.State==0)&(test.CosSim < thre)])
            # print(thre)
            # print(len(test[(test.State==1)&(test.CosSim >= thre)]))
            # print(len(test[(test.State==0)&(test.CosSim < thre)]))
            acu_queue.append(c/len(test))

        acu=sum(acu_queue)/(16-int(timelength/60))
        xaxis.append(timelength/60)
        acu_all.append(acu)



    import plotly.graph_objects as go

    # fig2 = go.Figure()
    # fig2.add_trace(go.Scatter(x=xaxis, y=acu_all,
    #                 mode='lines+markers',
    #                 name='WiTrack'))
    # fig2.update_layout(title='TranData Size Comparison', xaxis = dict(title='Traindata Size (minutes)'),  yaxis = dict(title='Accuracy'), font = dict(family='Montserrat SemiBold',size = 14))
    # pio.write_image(fig2, 'chart/TranData Size Comparison(WiTrack)_v3.png', scale=2)
    # fig2.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xaxis, y=acu_all,
                    mode='lines+markers',
                    name='WiTrack'))
    fig.add_trace(go.Scatter(x=xaxis, y=yaxis,
                    mode='lines+markers',
                    name='With CosSimularity'))
    fig.add_trace(go.Scatter(x=xaxis_without, y=yaxis_without,
                    mode='lines+markers',
                    name='Without CosSimularity'))
    fig.update_layout(title='Trandata Size Comparison', xaxis = dict(title='Traindata Size (minutes)'),  yaxis = dict(title='Accuracy'), font = dict(family='Courier New, monospace',size = 18,color="#7f7f7f"))
    pio.write_image(fig, 'chart/TranData Size Comparison(All)_v3.png', scale=2)
    fig.show()





# # ========== Tree Depth Effect ==========

#     temp=[]
#     temp2=[]
#     for treedep in range(1,50):
#         TrainData = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
#         TrainData=TrainData.set_index('pair')
#         # TrainData1211 = pd.read_csv('TrainData1211.csv', error_bad_lines=False)
#         # print(TrainData1211)

#         # print(TrainData)

#         X = TrainData.iloc[:, 0:13]
#         y = TrainData.iloc[:, 13]
#         # print(X)

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#         dtc = DTC(max_depth=treedep)
#         # dtc = fit_model_k_fold(X_train, y_train)
#         dtc.fit(X_train, y_train)

#         # print(dtc)

#         # # 特徵重要度 
#         # print(dtc.feature_importances_)
#         predict_target = dtc.predict(X_test)
#         # print(predict_target)
#         # print(sum(predict_target == y_test))
        
#         print('Accuracy：', dtc.score(X_test, y_test))
#         temp.append(dtc.score(X_test, y_test))
#         temp2.append(treedep)

#     import plotly.graph_objects as go
#     fig = go.Figure()
#     # fig.add_trace(go.Scatter(x=xaxis, y=acu_all,
#     #                 mode='lines+markers',
#     #                 name='WiTrack'))
#     fig.add_trace(go.Scatter(x=temp2, y=temp,
#                     mode='lines+markers'))
#     fig.update_layout(title='Effect of Tree Depth', xaxis = dict(title='Tree Depth'),  yaxis = dict(title='Accuracy'), font = dict(family='Montserrat SemiBold',size = 14))
#     pio.write_image(fig, 'chart/Tree Depth Effect.png', scale=2)
#     fig.show()

