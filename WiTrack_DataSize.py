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

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xaxis, y=yaxis,
                    mode='lines+markers',
                    name='With CosSimularity'))
    fig.update_layout(title='TranData Size Comparison', xaxis = dict(title='Traindata Size (minutes)'),  yaxis = dict(title='Accuracy'), font = dict(family='Montserrat SemiBold',size = 14))
    pio.write_image(fig, 'chart/TranData Size Comparison_v3.png', scale=2)
    fig.show()