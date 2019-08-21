import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio

# #! Single shift chart
# dfcsv = pd.read_csv("./follower/follower_1000_UE15.csv")

# #* single shift 0~5sec
# data = [go.Scatter(
#     # x=list(range(0,30000,1000)),
#     y=dfcsv[dfcsv.columns[0]],
#     marker = dict(color='rgb(80,140,191)'),
#     # line=dict(shape='spline'),
#     name='UE1 - UE2',
# )]
# layout = go.Layout(
#     xaxis = dict(
#         title='Time Shift (sec)',
#         range=[0, 5],
#         showline = True,
#         ),
#     yaxis = dict(
#         title='Similarity',
#         range=[0.6, 0.8],
#         showline = True,
#         ),
#     font = dict(family='Montserrat SemiBold', size=14),
#     # paper_bgcolor='rgba(0,0,0,0)',
#     # plot_bgcolor='rgba(0,0,0,0)',
#     showlegend=True,
#     annotations=[
#         dict(
#             x=dfcsv[dfcsv.columns[0]].head(6).idxmax(),
#             y=dfcsv[dfcsv.columns[0]].head(6).max(),
#             xref='x',
#             yref='y',
#             text=dfcsv[dfcsv.columns[0]].head(6).max().round(4),
#             showarrow=True,
#             arrowhead=5,
#             ax=0,
#             ay=-30
#         )
#     ],
# )
# fig = go.Figure(data=data, layout=layout)
# # py.offline.plot(fig, filename='./follower.html')
# pio.write_image(fig, 'image/timeshift5.png', scale=2)

# #* single shift 5~25sec
# layout = go.Layout(
#     xaxis = dict(
#         title='Time Shift (sec)',
#         range=[5, 25],
#         showline = True,
#         ),
#     yaxis = dict(
#         title='Similarity',
#         range=[0.6, 0.8],
#         showline = True,
#         ),
#     font = dict(family='Montserrat SemiBold', size=14),
#     showlegend=True,
#     annotations=[
#         dict(
#             x=dfcsv[dfcsv.columns[0]].idxmax(),
#             y=dfcsv[dfcsv.columns[0]].max(),
#             xref='x',
#             yref='y',
#             text=dfcsv[dfcsv.columns[0]].max().round(4),
#             showarrow=True,
#             arrowhead=5,
#             ax=0,
#             ay=-30
#         )
#     ],
#     # paper_bgcolor='rgba(0,0,0,0)',
#     # plot_bgcolor='rgba(0,0,0,0)'
# )
# fig = go.Figure(data=data, layout=layout)
# # py.offline.plot(fig, filename='./follower.html')
# pio.write_image(fig, 'image/timeshift25.png', scale=2)

# #* single shift 0~25sec
# layout = go.Layout(
#     xaxis = dict(
#         title='Time Shift (sec)',
#         range=[0, 25],
#         showline = True,
#         ),
#     yaxis = dict(
#         title='Similarity',
#         range=[0.6, 0.8],
#         showline = True,
#         ),
#     font = dict(family='Montserrat SemiBold', size=14),
#     showlegend=True,
#     annotations=[
#         dict(
#             x=dfcsv[dfcsv.columns[0]].idxmax(),
#             y=dfcsv[dfcsv.columns[0]].max(),
#             xref='x',
#             yref='y',
#             text=dfcsv[dfcsv.columns[0]].max().round(4),
#             showarrow=True,
#             arrowhead=5,
#             ax=0,
#             ay=-30
#         )
#     ],
#     # paper_bgcolor='rgba(0,0,0,0)',
#     # plot_bgcolor='rgba(0,0,0,0)'
# )
# fig = go.Figure(data=data, layout=layout)
# # py.offline.plot(fig, filename='./follower.html')
# pio.write_image(fig, 'image/timeshift_all.png', scale=2)


#! Mutiple shift chart
df1 = pd.read_csv("./follower/follower_1000_UE15.csv")
df2 = pd.read_csv("./follower/follower_1000_UE16.csv")
df3 = pd.read_csv("./follower/follower_1000_UE36.csv")
df4 = pd.read_csv("./follower/follower_1000_UE45_alter2.csv")
df5 = pd.read_csv("./follower/follower_1000_UE35.csv")
df6 = pd.read_csv("./follower/follower_1000_UE46.csv")

# df1 = pd.read_csv("./aa")
# df2 = pd.read_csv("./bb")
# df3 = pd.read_csv("./cc")
# df4 = pd.read_csv("./dd")
# df5 = pd.read_csv("./ee")
# df6 = pd.read_csv("./ff")

def getShiftScatter(df, name, color):
    trace = go.Scatter(
            y=df[df.columns[0]],
            # marker = dict(color=color),
            # line=dict(shape='spline'),
            mode='lines+markers',
            name=name,
        )
    return trace

def getShiftAllScatter(df, name):
    trace = go.Scatter(
            # y=df[df.columns[0]],
            y=df,
            # marker = dict(color=color),
            # line=dict(shape='spline'),
            mode='lines+markers',
            name=name,
        )
    return trace

# #* shift similarity compared with others
# df1 = pd.read_csv('./shift/wifi/20/Perry.csv')
# # print(df1)
# # print(df1.index.name)
# # print(df1.columns[0])
# data = []
# for column in df1:
#     if column == df1.columns[0]:
#         pass
#     else:
#         data.append(getShiftAllScatter(df1[column], column))

# layout = go.Layout(
#     xaxis = dict(
#         title='Time Shift (sec)',
#         range=[0, 30],
#         showline = True,
#         ),
#     yaxis = dict(
#         title='Similarity',
#         # range=[0.3, 1.0],
#         showline = True,
#         ),
#     font = dict(family='Montserrat SemiBold', size=14),
# )

# fig = go.Figure(data=data, layout=layout)
# py.offline.plot(fig, filename='./shift_man.html')

# #* timeshift_mult_5.png
# data = [
#     getShiftScatter(df1, 'Single', 'rgb(112,193,71)'),
#     getShiftScatter(df2, 'Follower1_1', 'rgb(110,95,169)'),
#     getShiftScatter(df3, 'Follower1_2', 'rgb(75,172,198)'),
#     getShiftScatter(df4, 'Companion', 'rgb(75,90,198)'),
#     getShiftScatter(df5, 'Follower2_1','rgb(80,10,198)'),
#     getShiftScatter(df6, 'Loner', 'rgb(75,172,198)'),
# ]

# layout = go.Layout(
#     xaxis = dict(
#         title='Time Shift (sec)',
#         range=[0, 30],
#         showline = True,
#         ),
#     yaxis = dict(
#         title='Similarity',
#         # range=[0.6, 0.8],
#         showline = True,
#         ),
#     font = dict(family='Montserrat SemiBold', size=14),
#     # legend=dict(orientation="h"),
#     # paper_bgcolor='rgba(0,0,0,0)',
#     # plot_bgcolor='rgba(0,0,0,0)',
# )


# fig = go.Figure(data=data, layout=layout)
# py.offline.plot(fig, filename='./follower.html')
# pio.write_image(fig, 'image/timeshift_mult_5.png', scale=2)

# #* timeshift_mult_5.png
data = [
    # getShiftScatter(df1, 'UE1 - UE5', 'rgb(112,193,71)'),
    # getShiftScatter(df2, 'UE1 - UE6', 'rgb(110,95,169)'),
    # getShiftScatter(df3, 'UE3 - UE6', 'rgb(75,172,198)'),
    getShiftScatter(df2, '(UE1 , UE6)', 'rgb(110,95,169)'),
    getShiftScatter(df3, '(UE3 , UE6)', 'rgb(75,172,198)'),
    # getShiftScatter(df4, 'UE4 - UE5'),
    # getShiftScatter(df5, 'UE3 - UE5'),
    # getShiftScatter(df6, 'UE4 - UE6', 'rgb(75,172,198)'),
]

layout = go.Layout(
    xaxis = dict(
        title='Time Shift (sec)',
        range=[0, 5],
        showline = True,
        ),
    yaxis = dict(
        title='Similarity',
        range=[0.6, 0.8],
        showline = True,
        ),
    font = dict(family='Montserrat SemiBold', size=14),
    # legend=dict(orientation="h"),
    # paper_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor='rgba(0,0,0,0)',
)


fig = go.Figure(data=data, layout=layout)
# py.offline.plot(fig, filename='./follower.html')
pio.write_image(fig, 'image/timeshift_mult_5.png', scale=2)

# #* timeshift_mult_25.png
layout = go.Layout(
    xaxis = dict(
        title='Time Shift (sec)',
        range=[5, 25],
        showline = True,
        ),
    yaxis = dict(
        title='Similarity',
        range=[0.6, 0.8],
        showline = True,
        ),
    font = dict(family='Montserrat SemiBold', size=14),
    # legend=dict(orientation="h"),
    annotations=[
        # dict(
        #     x=df1[df1.columns[0]].idxmax(),
        #     y=df1[df1.columns[0]].max(),
        #     xref='x',
        #     yref='y',
        #     # text=df1[df1.columns[0]].max().round(4),
        #     text='('+str(df1[df1.columns[0]].idxmax())+','+str(df1[df1.columns[0]].max().round(4))+')',
        #     showarrow=True,
        #     arrowhead=5,
        #     ax=0,
        #     ay=-40
        # ),
        dict(
            x=df2[df2.columns[0]].idxmax(),
            y=df2[df2.columns[0]].max(),
            xref='x',
            yref='y',
            # text=df2[df2.columns[0]].max().round(4),
            text='('+str(df2[df2.columns[0]].idxmax())+','+str(df2[df2.columns[0]].max().round(4))+')',
            showarrow=True,
            arrowhead=5,
            ax=0,
            ay=-40
        ),
        dict(
            x=df3[df3.columns[0]].idxmax(),
            y=df3[df3.columns[0]].max(),
            xref='x',
            yref='y',
            # text=df3[df3.columns[0]].max().round(4),
            text='('+str(df3[df3.columns[0]].idxmax())+','+str(df3[df3.columns[0]].max().round(4))+')',
            showarrow=True,
            arrowhead=5,
            ax=0,
            ay=-50
        ),
        # dict(
        #     x=df4[df4.columns[0]].idxmax(),
        #     y=df4[df4.columns[0]].max(),
        #     xref='x',
        #     yref='y',
        #     text=df4[df4.columns[0]].max().round(4),
        #     showarrow=True,
        #     arrowhead=5,
        #     ax=0,
        #     ay=-30
        # )
    ],
    # paper_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor='rgba(0,0,0,0)'
)
fig = go.Figure(data=data, layout=layout)
# py.offline.plot(fig, filename='./follower.html')
pio.write_image(fig, 'image/timeshift_mult_25.png', scale=2)

# #* timeshift_mult_all.png
# layout = go.Layout(
#     xaxis = dict(
#         title='Time Shift (sec)',
#         range=[0, 25],
#         showline = True,
#         ),
#     yaxis = dict(
#         title='Similarity',
#         range=[0.6, 0.8],
#         showline = True,
#         ),
#     font = dict(family='Montserrat SemiBold', size=14),
#     # legend=dict(orientation="h"),
#     # annotations=[
#     #     dict(
#     #         x=dfcsv[dfcsv.columns[0]].idxmax(),
#     #         y=dfcsv[dfcsv.columns[0]].max(),
#     #         xref='x',
#     #         yref='y',
#     #         text=dfcsv[dfcsv.columns[0]].max().round(4),
#     #         showarrow=True,
#     #         arrowhead=5,
#     #         ax=0,
#     #         ay=-30
#     #     )
#     # ],
#     # paper_bgcolor='rgba(0,0,0,0)',
#     # plot_bgcolor='rgba(0,0,0,0)'
#     annotations=[
#         # dict(
#         #     x=df1[df1.columns[0]].idxmax(),
#         #     y=df1[df1.columns[0]].max(),
#         #     xref='x',
#         #     yref='y',
#         #     text=df1[df1.columns[0]].max().round(4),
#         #     showarrow=True,
#         #     arrowhead=5,
#         #     ax=0,
#         #     ay=-40
#         # ),
#         dict(
#             x=df2[df2.columns[0]].idxmax(),
#             y=df2[df2.columns[0]].max(),
#             xref='x',
#             yref='y',
#             text=df2[df2.columns[0]].max().round(4),
#             showarrow=True,
#             arrowhead=5,
#             ax=0,
#             ay=-40
#         ),
#         dict(
#             x=df3[df3.columns[0]].idxmax(),
#             y=df3[df3.columns[0]].max(),
#             xref='x',
#             yref='y',
#             text=df3[df3.columns[0]].max().round(4),
#             showarrow=True,
#             arrowhead=5,
#             ax=0,
#             ay=-40
#         ),
#         # dict(
#         #     x=df4[df4.columns[0]].idxmax(),
#         #     y=df4[df4.columns[0]].max(),
#         #     xref='x',
#         #     yref='y',
#         #     text=df4[df4.columns[0]].max().round(4),
#         #     showarrow=True,
#         #     arrowhead=5,
#         #     ax=0,
#         #     ay=-30
#         # ),
#         # dict(
#         #     x=df5[df5.columns[0]].idxmax(),
#         #     y=df5[df5.columns[0]].max(),
#         #     xref='x',
#         #     yref='y',
#         #     text=df5[df5.columns[0]].max().round(4),
#         #     showarrow=True,
#         #     arrowhead=5,
#         #     ax=0,
#         #     ay=-30
#         # )
#     ],
# )
# fig = go.Figure(data=data, layout=layout)
# # py.offline.plot(fig, filename='./follower.html')
# pio.write_image(fig, 'image/timeshift_mult_all.png', scale=2)

# #* basic shift plot
# dfcsv = pd.read_csv("./follower.csv")
# print(dfcsv[dfcsv.columns[0]])
# trace = go.Scatter(
#     # x=list(range(0,30000,1000)),
#     y=dfcsv[dfcsv.columns[0]],
#     line=dict(
#         shape='spline'
#     )
# )
# data = [trace]
# py.offline.plot(data, filename='./follower.html')