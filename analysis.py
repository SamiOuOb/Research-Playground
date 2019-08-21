import pandas as pd

import csv
from itertools import combinations, permutations
from plotly import tools
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import matplotlib.pyplot as plt
# pd.options.mode.chained_assignment = None

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

# scenario = 'All in all'
# start_time = '11:47:00'
# end_time = '12:41:00'

# scenario = 'Two Guys with Two Phones'
# start_time = '11:47:00'
# end_time = '11:59:30'

# scenario = 'A Guy with Two Phones'
# start_time = '12:00:00'
# end_time = '12:13:25'

# scenario = 'One Guy after another'
# start_time = '12:14:30'
# end_time = '12:25:50'

# scenario = 'Opposite Direction'
# start_time = '12:30:30'
# end_time = '12:41:00'

# device_1_name = 'HTC'
# device_1_mac = '40:4e:36:08:20:d1'

# device_1_name = 'Asus 6'
# device_1_mac = '10:c3:7b:2e:e0:00'

# device_2_name = 'Asus L'
# device_2_mac = '00:0a:f5:ce:d6:e4'

# device_2_name = 'ASUS'
# device_2_mac = '0c:9d:92:72:f9:3a'

# device_2_name = 'Asus S'
# device_2_mac = '14:dd:a9:bc:11:0c'

# device_2_name = 'Ipad M'
# device_2_mac = 'e0:f5:c6:47:94:60'


scenario = 'All'
start_time = '16:30:00'
end_time = '16:45:00'

# device_1_name = 'HTC'
# # device_1_mac = '4E:60:38:4A:2A:54'
# device_1_mac = '50:6A:A5:C5:91:85'

# device_1_name = 'MI1/Jeff'
# device_1_mac = 'fb:7a:8c:09:78:3a'

# device_1_name = 'ibeacon/Mack'
# device_1_mac = '12:3b:6a:1a:62:ef'

device_1_name = 'ibeacon/Han'
device_1_mac = '12:3b:6a:1a:75:71'

# device_2_name = 'ibeacon/Tony'
# device_2_mac = '12:3b:6a:1a:62:ed'

device_2_name = 'ibeacon/Perry'
device_2_mac = '12:3b:6a:1a:75:5b'

# device_2_name = 'ibeacon/Perry'
# device_2_mac = '12:3b:6a:1a:75:5a'

# device_2_name = 'ibeacon/Sami'
# device_2_mac = '12:3b:6a:1a:75:66'

# device_2_name = 'ibeacon/Jeff'
# device_2_mac = '12:3b:6a:1a:75:61'

# device_2_name = 'MI2/Tony'
# device_2_mac = 'd4:15:fd:d7:62:20'


# device_2_name = 'MI3/Perry'
# device_2_mac = 'cd:fc:cc:d0:f5:25'

# device_2_name = 'ASUS'
# # device_2_mac = '44:CB:09:BF:4B:7A'
# device_2_mac = '73:57:68:36:EB:2F'
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

mib = {
    'Jeff'   : 'd4:15:fd:d7:62:20', # Only 11/27
    'Perry'  : 'cd:fc:cc:d0:f5:25',
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


interval = 5
alpha = 0.5

# file0 = "./1005/20181005_100.log"
# file1 = "./1005/20181005_101.log"
# file2 = "./1005/20181005_102.log"
# csvfilename = "./follower/follower_1000_UE15_test.csv"
file0 = "./1127/ble/20181127_100.log"
file1 = "./1127/ble/20181127_101.log"
file2 = "./1127/ble/20181127_102.log"

file3 = "./1127/wifi/20181127_100.log"
file4 = "./1127/wifi/20181127_101.log"
file5 = "./1127/wifi/20181127_102.log"

def getRSSI(filename, ble=False):
    df = pd.read_csv(filename, sep='\t', parse_dates=[0], error_bad_lines=False)
    if 'ble' in filename:
        df.columns = ['time','mac','type','rssi','uuid']
        rate='1'
    else:
        df.columns = ['time','mac','chip','ap','rssi']
        rate='2'
    df = df.set_index('time')
    df = df.between_time(start_time, end_time)
    # print(device_1_mac, device_2_mac)
    if ble == True:
        probe1 = df[df.uuid == device_1_mac].resample(rate+'S').agg(dict(rssi='max')).dropna()
        probe2 = df[df.uuid == device_2_mac].resample(rate+'S').agg(dict(rssi='max')).dropna()
    else:
        probe1 = df[df.mac == device_1_mac].resample(rate+'S').agg(dict(rssi='max')).dropna()
        probe2 = df[df.mac == device_2_mac].resample(rate+'S').agg(dict(rssi='max')).dropna()
    probe1.rssi = getNormalize(probe1.rssi)
    probe2.rssi = getNormalize(probe2.rssi)
    probe1 = getSES(probe1, alpha, False)
    probe2 = getSES(probe2, alpha, False)
    return probe1, probe2

def getShiftRSSI(filename, interval, shift, ble=False):
    probe1, probe2 = getRSSI(filename, ble=ble)
    probe2 = probe2.shift(shift, freq='S')
    mean1 = probe1.groupby(pd.Grouper(freq=str(interval)+'S')).rssi.mean()
    mean2 = probe2.groupby(pd.Grouper(freq=str(interval)+'S')).rssi.mean()
    mean = pd.concat([mean1, mean2], axis=1)
    return mean

def getCount(filename, ble=False):
    df = pd.read_csv(filename, sep='\t', parse_dates=[0], error_bad_lines=False)
    if 'ble' in filename:
        df.columns = ['time','mac','type','rssi','uuid']
    else:
        df.columns = ['time','mac','chip','ap','rssi']
    df = df.set_index('time')
    df = df.between_time(start_time, end_time)
    if ble == True:
        probe1 = df[df.uuid == device_1_mac]
    else:
        probe1 = df[df.mac == device_1_mac]
    return probe1.rssi.count()

def getNormalize(column):
    column = ( column - column.min() ) / ( column.max() - column.min() )
    return column


def getCosSim(m0, m1, m2):
    rssi_info = m0.append([m1, m2], ignore_index=True)
    rssi_info.columns = ['dev1','dev2']
    a = rssi_info.dev1.fillna(0).tolist()
    b = rssi_info.dev2.fillna(0).tolist()
    cs = 1 - cosine(a, b)
    return cs

def getEuclidean(m0, m1, m2):
    rssi_info = m0.append([m1, m2], ignore_index=True)
    rssi_info.columns = ['dev1','dev2']
    a = rssi_info.dev1.fillna(0).tolist()
    b = rssi_info.dev2.fillna(0).tolist()
    eu = euclidean(a, b)
    return eu

def getMacro(file0, file1, file2, interval=interval, shift=0, ble=False):
    mean_0 = getShiftRSSI(file0, interval, shift, ble=ble)
    mean_1 = getShiftRSSI(file1, interval, shift, ble=ble)
    mean_2 = getShiftRSSI(file2, interval, shift, ble=ble)
    cs_0 = getCosSim(mean_0, mean_1, mean_2)
    interval = interval/2
    mean_0 = getShiftRSSI(file0, interval, shift, ble=ble)
    mean_1 = getShiftRSSI(file1, interval, shift, ble=ble)
    mean_2 = getShiftRSSI(file2, interval, shift, ble=ble)
    cs_1 = getCosSim(mean_0[::2], mean_1[::2], mean_2[::2])
    cs_2 = getCosSim(mean_0[1::2], mean_1[1::2], mean_2[1::2])
    return (alpha*cs_0) + ((1-alpha)*(cs_1+cs_2)/2)

def getSES(probe, alpha=None, auto=True):
    # ses=sum([alpha * (1 - alpha) ** i * x for i, x in 
    #             enumerate(reversed(probe.rssi))])
    fit = SimpleExpSmoothing(probe.rssi).fit(smoothing_level=alpha, optimized=auto)
    ses = fit.fittedvalues.shift(-1)
    ses = pd.DataFrame(ses).reset_index()
    ses.columns = ['time','rssi']
    ses = ses.set_index('time')
    return ses

def getShift(file0, file1, file2, interval, ble=False):
    shift_list = []
    for i in range(0,-31,-1):
        shift_list.append(getMacro(file0,file1,file2,interval,i,ble=ble))
    return shift_list

def getShiftcsv(file0, file1, file2, interval, csvfilename='test.csv', ble=False):
    shift_list = []
    for i in range(0,-31,-1):
        shift_list.append(getMacro(file0,file1,file2,interval,i,ble=ble))
    with open(csvfilename, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in shift_list:
            writer.writerow([val]) 
    pass


# df = pd.DataFrame(columns=['name','sim'])
# # * Similarity CSV

def getShiftall(dev_dict, mode, ble=False):
    for x in dev_dict:
        # ls=[]
        # df = pd.DataFrame()
        dt = {}
        device_1_name = x
        device_1_mac = dev_dict[x]    
        for y in dev_dict:
            if y == x:
                pass
            else:
                device_2_name = y
                device_2_mac = dev_dict[y]
                # dt.update({y:getShift(file0, file1, file2, interval, ble=ble)})
                print(device_1_mac,device_2_mac)
                # df.append({y: getShift(file0, file1, file2, interval, ble=ble)}, ignore_index=True)
                # ls.append([h+'_'+k] + getShift(file0, file1, file2, interval, ble=ble))
        # df = pd.DataFrame(ls,columns=['name']+[str(x) for x in range(0,31,1)])
        # df = df.set_index('name')
        df = pd.DataFrame(dt)
        df.to_csv('./shift/'+mode+'/'+x+'.csv', index_label='shift')
        dt.clear()
    pass
# device_1_mac = ibeacon['Jeff']
# device_2_mac = ibeacon['Han']
# print(getShift(file0, file1, file2, interval))
# getShiftall(ibeacon, 'ibeacon')
# print(device_1_mac,device_2_mac)
# getShiftall(mib, 'mib')
# getShiftall(ble, 'ble')
# getShiftall(wifi, 'wifi', ble=False)

# for x in ibeacon:
#     dt = {}
#     device_1_name = x
#     device_1_mac = ibeacon[x]    
#     for y in ibeacon:
#         if y == x:
#             pass
#         else:
#             device_2_name = y
#             device_2_mac = ibeacon[y]
#             dt.update({y:getShift(file0, file1, file2, interval)})
#     df = pd.DataFrame(dt)
#     df.to_csv('./shift/ibeacon/'+x+'.csv', index_label='shift')

# for x in ble:
#     dt = {}
#     device_1_name = x
#     device_1_mac = ble[x]    
#     for y in ble:
#         if y == x:
#             pass
#         else:
#             device_2_name = y
#             device_2_mac = ble[y]
#             dt.update({y:getShift(file0, file1, file2, interval,ble=True)})
#     df = pd.DataFrame(dt)
#     df.to_csv('./shift/ble/'+x+'.csv', index_label='shift')

# for x in mib:
#     dt = {}
#     device_1_name = x
#     device_1_mac = mib[x]    
#     for y in mib:
#         if y == x:
#             pass
#         else:
#             device_2_name = y
#             device_2_mac = mib[y]
#             dt.update({y:getShift(file0, file1, file2, interval)})
#     df = pd.DataFrame(dt)
#     df.to_csv('./shift/mib/'+x+'.csv', index_label='shift')
interval = 10 

for x in wifi:
    dt = {}
    device_1_name = x
    device_1_mac = wifi[x]    
    for y in wifi:
        if y == x:
            pass
        else:
            device_2_name = y
            device_2_mac = wifi[y]
            dt.update({y:getShift(file3, file4, file5, interval)})
    df = pd.DataFrame(dt)
    df.to_csv('./shift/wifi/10/'+x+'.csv', index_label='shift')



# dfs=[]
# for h,k in combinations(ibeacon,2):
#     # print(ibeacon[h],ibeacon[k])
#     device_1_name = h
#     device_1_mac = ibeacon[h]
#     device_2_name = k
#     device_2_mac = ibeacon[k]
#     # dff.append((h+'_'+k, getMacro(file0,file1,file2,interval,0)))
#     # df.append(pd.Series([h+'_'+k, getMacro(file0,file1,file2,interval,0)]),ignore_index=True)
#     # df.append({'name':h+'_'+k, 'sim':getMacro(file0,file1,file2,interval,0)},ignore_index=True)
#     # df.append({'name':'test', 'sim':getMacro(file0,file1,file2,interval,0)},ignore_index=True)
#     # dfs.append([h+'_'+k, getMacro(file0,file1,file2)])
#     # getShiftcsv(file0, file1, file2, interval, './shift/'+h+'_'+k+'.csv')
#     # df = getShiftcsv(file0, file1, file2, interval)
#     dfs.append([h+'_'+k] + getShift(file0, file1, file2, interval))
# # df.append(dff,ignore_index=True)
# # df = pd.DataFrame(dfs,columns=['name','sim'])
# # df = df.set_index('name')
# # df.to_csv('sim_beacon.csv')
# print(dfs)
# df = pd.DataFrame(dfs,columns=['name']+[str(x) for x in range(0,31,1)])
# df = df.set_index('name')
# df.to_csv('beacon.csv')

# for x in ibeacon:
#     device_2_mac = ibeacon[x]
#     for y in ibeacon:
#         if y == x:
#             pass
#         else:
#             device_2_mac = ibeacon[y]
#             print(device_1_mac, device_2_mac,getMacro(file0, file1, file2))
#     print(x)
# print(device_1_mac, device_2_mac)
# hk=list(permutations(ibeacon,2))
# print(hk)
    # print(h,k)
    # for h in h,k:
    #     print(h,k)
    # print('x')


# dfs=[]
# for h,k in combinations(ibeacon,2):
#     # print(ibeacon[h],ibeacon[k])
#     device_1_name = h
#     device_1_mac = ibeacon[h]
#     device_2_name = k
#     device_2_mac = ibeacon[k]
#     # dff.append((h+'_'+k, getMacro(file0,file1,file2,interval,0)))
#     # df.append(pd.Series([h+'_'+k, getMacro(file0,file1,file2,interval,0)]),ignore_index=True)
#     # df.append({'name':h+'_'+k, 'sim':getMacro(file0,file1,file2,interval,0)},ignore_index=True)
#     # df.append({'name':'test', 'sim':getMacro(file0,file1,file2,interval,0)},ignore_index=True)
#     # dfs.append([h+'_'+k, getMacro(file0,file1,file2)])
#     # getShiftcsv(file0, file1, file2, interval, './shift/'+h+'_'+k+'.csv')
#     # df = getShiftcsv(file0, file1, file2, interval)
#     dfs.append([h+'_'+k] + getShift(file0, file1, file2, interval))
# # df.append(dff,ignore_index=True)
# # df = pd.DataFrame(dfs,columns=['name','sim'])
# # df = df.set_index('name')
# # df.to_csv('sim_beacon.csv')
# print(dfs)
# df = pd.DataFrame(dfs,columns=['name']+[str(x) for x in range(0,31,1)])
# df = df.set_index('name')
# df.to_csv('beacon.csv')

# print([['as'] + getShift(file0, file1, file2, interval)])
list = [1,2,3]
a = 0
print([a]+list)
# dfs=[]
# for h,k in combinations(wifi,2):
#     device_1_name = h
#     device_1_mac = wifi[h]
#     device_2_name = k
#     device_2_mac = wifi[k]
#     dfs.append([h+'_'+k, getMacro(file3,file4,file5,interval,0)])
#     getShiftcsv(file3, file4, file5, interval, './shift/'+h+'_'+k+'.csv')
# df = pd.DataFrame(dfs,columns=['name','sim'])
# df = df.set_index('name')
# df.to_csv('sim_wifi.csv')

# dfs=[]
# for h,k in combinations(ble,2):
#     device_1_name = h
#     device_1_mac = ble[h]
#     device_2_name = k
#     device_2_mac = ble[k]
#     dfs.append([h+'_'+k, getMacro(file0,file1,file2,interval,0)])
#     getShiftcsv(file0, file1, file2, interval, './shift/'+h+'_'+k+'.csv',ble=1)
# df = pd.DataFrame(dfs,columns=['name','sim'])
# df = df.set_index('name')
# df.to_csv('sim_ble.csv')

## * Count CSV
# dfc=[]
# for h in ibeacon:
#     device_1_name = h
#     device_1_mac = ibeacon[h]
#     # count = [getCount(file0), getCount(file1), getCount(file2)]
#     # # dfc.append([h, count, sum(count)])
#     # dfc.append([h])
#     # dfc.extend([count, sum(count)])
#     a, b, c = getCount(file0), getCount(file1), getCount(file2)
#     dfc.append([h, a, b, c, a+b+c])
#     # dfc.append([h, getCount(file0), getCount(file1), getCount(file2), getCount(file0)+getCount(file1)+getCount(file2)])
# # print(dfc)
# df = pd.DataFrame(dfc, columns=['name','count0','count1','count2','total'])
# df = df.set_index('name')
# df.to_csv('count_beacon.csv')#, index_label="name")
# # df.to_csv('count_beacon.csv', index_label="index_label")
# aa = pd.read_csv('count_beacon.csv')
# print(aa.name)

# dfc=[]
# for h in wifi:
#     device_1_name = h
#     device_1_mac = wifi[h]
#     a, b, c = getCount(file3), getCount(file4), getCount(file5)
#     dfc.append([h, a, b, c, a+b+c])
    
# # print(dfc)
# df = pd.DataFrame(dfc, columns=['name','count0','count1','count2','total'])
# df = df.set_index('name')
# df.to_csv('count_wifi.csv')
# # a,b = getRSSI(file3)
# # print(a.count())

# dfc=[]
# for h in ble:
#     device_1_name = h
#     device_1_mac = ble[h]
#     a, b, c = getCount(file0,ble=1), getCount(file1,ble=1), getCount(file2,ble=1)
#     dfc.append([h, a, b, c, a+b+c])
# df = pd.DataFrame(dfc, columns=['name','count0','count1','count2','total'])
# df = df.set_index('name')
# df.to_csv('count_ble.csv')

# print(getRSSI(file0))
# getShiftcsv(file0, file1, file2, interval, 'ff')
# ax = [1,2,3,4,5]
ay = {
    'a':'1',
    'b':'2',
    'c':'3',
}
# print(len(list(combinations(ax, 2))))
# print(dict(combinations(ay, 2)))
# az=[(k,h,ay(k),ay(h)) for k,h in combinations(ay, 2)]
# print(az)

for k,h in combinations(ay, 2):
    # print(ay[k], ay[h])
    print(k,h)
# for i in range(len(ay)):
#     for j in range(i+1,len(ay)):
#         print(ay[i],ay[j])


# dev1_0, dev2_0 = getRSSI("./1122/ble/20181122_100.log")

# dev1_0, dev2_0 = getRSSI("./1005/20181005_100.log")
# dev1_1, dev2_1 = getRSSI("./1005/20181005_101.log")
# dev1_2, dev2_2 = getRSSI("./1005/20181005_102.log")

# fit3 = SimpleExpSmoothing(dev2_0.rssi).fit()
# fcast3 = fit3.forecast(12)
# # print(fit3)
# # print(fcast3)
# # fit3.plot()
# # fcast3.plot()
# dev2_0.rssi.plot(marker='o', color='red')
# fit3.fittedvalues.shift(-1).plot(marker='o', color='green')
# # print(fit3.fittedvalues)
# print(fit3.fittedvalues.shift(-1))
# plt.show()

# print(dev1_0.rssi.count())
# model = SimpleExpSmoothing(dev1_0.rssi)
# model_fit = model.fit()
# guess = model_fit.predict(len(dev1_0.rssi), len(dev1_0.rssi))
# print(guess)




