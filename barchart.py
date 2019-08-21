import plotly as py
import plotly.graph_objs as go
import plotly.io as pio

# pio.orca.config.executable = '/home/forged05/miniconda3/bin/orca'

trace1 = go.Bar(
    # x = ['Hand-Hand', 'Hand-Pocket', 'Hand-Backpack','Pocket-Backpace'],
    # x = ['UE1 - UE2', 'UE1 - UE3', 'UE1 - UE4', 'UE3 - UE4'],
    # x = ['(UE1,UE2)', '(UE1,UE3)', '(UE1,UE4)', '(UE3,UE4)'],
    x = ['(UE1 , UE2)', '(UE1 , UE3)', '(UE1 , UE4)', '(UE3 , UE4)'],
    # y = [ 0.7398293076825063, 0.7608514911315144, 0.7316228844643318, 0.721375818521713 ],
    y = [ 0.7225117945178394, 0.7917993870463852,  0.7522955843918352, 0.7237880842427684 ],
    text = [ 0.7225, 0.7917,  0.7522, 0.7237 ],
    textposition='outside',
    # 0.7998660303452516
    width = [0.3, 0.3, 0.3, 0.3],
    # name = 'Normalized & Deleting ',
    name = 'Single',
    # marker = dict(color='rgb(80,140,191)'),
    marker = dict(color='rgb(75,172,198)'),
)

layout = go.Layout(
            barmode = 'overlay',
            font = dict(family='Montserrat SemiBold',size = 14),
            showlegend=True,
            yaxis = dict(title='Similarity'),
            legend=dict(orientation="h"),
         )
data = [trace1]
fig = go.Figure(data=data, layout=layout)
# py.offline.plot(fig, filename='./self.html')
pio.write_image(fig, 'image/one_guy_many_phones.png', scale=2)


trace1 = go.Bar(
    # x = ['Hand-Hand', 'Pocket-Pocket','Hand-Pocket', 'Hand-Backpack'],
    # x = ['UE2 - UE1', 'UE4 - UE3','UE2 - UE3', 'UE2 - UE5'],
    # x = ['UE1 - UE2', 'UE1 - UE4','UE3 - UE4', 'UE2 - UE5'],
    # x = ['UE1 - UE5', 'UE1 - UE6','UE3 - UE6', 'UE4 - UE5'],
    # x = ['(UE1,UE5)', '(UE1,UE6)','(UE3,UE6)', '(UE4,UE5)'],
    x = ['(UE1 , UE5)', '(UE1 , UE6)','(UE3 , UE6)', '(UE4 , UE5)'],
    # y = [ 0.7472151183615909, 0.7931646910746303, 0.758584681430124, 0.684064530032832 ],
    # y = [ 0.7363234089752445, 0.8060976595513927, 0.8013600113397343, 0.6977186303317559 ],
    y = [ 0.7363234089752445, 0.7560892703830526, 0.8060976595513927, 0.6977186303317559],
    text = [ 0.7363, '0.7560', '0.8060', 0.6977 ],
    textposition='outside',
    # name = 'Scenario1',
    name = 'Companion',
    marker = dict(color='rgb(110,95,169)'),
)

trace2 = go.Bar(
    # x = ['Hand-Hand', 'Pocket-Pocket','Hand-Pocket', 'Hand-Backpack'],
    # x = ['UE2 - UE1', 'UE4 - UE3','UE2 - UE3', 'UE2 - UE5'],
    # x = ['UE1 - UE2', 'UE1 - UE4','UE3 - UE4', 'UE2 - UE5'],
    # x = ['UE1 - UE5', 'UE1 - UE6','UE3 - UE6', 'UE4 - UE5'],
    # x = ['(UE1,UE5)', '(UE1,UE6)','(UE3,UE6)', '(UE4,UE5)'],
    x = ['(UE1 , UE5)', '(UE1 , UE6)','(UE3 , UE6)', '(UE4 , UE5)'],
    # y = [ 0.653002944000948, 0.6764945057702203, 0.6677563552483294, 0.6576426435084425 ],
    # y = [ 0.652805746290133, 0.7015804760310312, 0.7148608339612443, 0.6416441685799967 ],
    y = [ 0.652805746290133, 0.6514366343640663, 0.7015804760310312, 0.6416441685799967],
    text = [ 0.6528, 0.6514, 0.7015, 0.6416 ],
    textposition='outside',
    # name = 'Scenario2',
    name = 'Follower',
    # marker = dict(color='rgb(80,140,191)'),
    marker = dict(color='rgb(75,172,198)'),
)
# trace3 = go.Bar(
#     x = ['Hand-Hand', 'Pocket-Pocket','Hand-Pocket', 'Hand-Backpack'],
#     # y = [ 0.18872271864393836, 0.15957549625182504, 0.158230245529889, 0.24284775091279717 ],
#     y = [ 0.18152479229937618, 0.1765343069065499, 0.31045135303105675, 0.22039887024351631 ],
#     # name = 'Scenario3',
#     name = 'Opposite direction',
#     marker = dict(color='rgb(112,193,71)'),
# )
layout = go.Layout(
            # title = '<b>Scenario 1 - Side by Side</b>',
            yaxis = dict(title='Similarity'),
            # xaxis = dict(title='Time'),
            barmode = 'group',
            font = dict(family='Montserrat SemiBold',size = 14),
            # bargap=0.15,
            bargroupgap=0.1,
            legend=dict(orientation="h")
         )
data = [trace1,trace2]
fig = go.Figure(data=data, layout=layout)
# py.offline.plot(fig, filename='./scenario.html')
pio.write_image(fig, 'image/scenario.png', scale=2)



# trace = go.Bar(
#     x = ['HTC', 'ASUS P','ASUS L','ASUS S', 'IPad'],
#     y = [ 626, 361, 412, 585, 196 ],
#     width = [0.6, 0.6, 0.6, 0.6, 0.6],
#     # marker = dict(color='rgb(80,140,191)'),
#     marker = dict(color='rgb(110,95,169)'),
#     # name = 'Scenario3',
# )

# layout = go.Layout(
#             barmode = 'group',
#             font = dict(family='Montserrat SemiBold',size = 14),
#          )
# data = [trace]
# fig = go.Figure(data=data, layout=layout)
# py.offline.plot(fig, filename='./count.html')


# trace2 = go.Bar(
#     x = ['Hand-Hand', 'Pocket-Pocket','Hand-Pocket', 'Hand-Backpack'],
#     y = [ 0.652805746290133, 0.7015804760310312, 0.7148608339612443, 0.6416441685799967 ],
#     width = [0.3, 0.3, 0.3, 0.3],
#     name = 'Raw',
# )

# trace1 = go.Bar(
#     x = ['Hand-Hand', 'Pocket-Pocket','Hand-Pocket', 'Hand-Backpack'],
#     y = [ 0.7704630232947084, 0.7015804760310312, 0.7148608339612443, 0.6416441685799967 ],
#     # 0.7998660303452516
#     width = [0.3, 0.3, 0.3, 0.3],
#     name = 'Normalized & Deleting ',
# )

# layout = go.Layout(
#             barmode = 'overlay',
#             font = dict(family='Montserrat SemiBold',size = 12),
#             legend=dict(orientation="h"),
#          )
# data = [trace1, trace2]
# fig = go.Figure(data=data, layout=layout)
# py.offline.plot(fig, filename='./timeshift.html')

