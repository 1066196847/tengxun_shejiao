# coding=utf-8
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import csv
import os
import pickle
import cPickle
from math import ceil
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from a1 import a11

'''
函数说明：用 train_merge_3 test_merge_3 里面所有的特征训练出来结果
'''
def first_model():
    # 现有训练集的正负比例非常不均，我先直接训练下看
    train_merge_3 = pd.read_csv('../data/merge_data/train_merge_3.csv')
    test_merge_3 = pd.read_csv('../data/merge_data/test_merge_3.csv')
    fea = list(train_merge_3.columns)
    fea.remove('label')
    # fea.remove('clickTime')
    fea.remove('conversionTime') # 特征列

    import xgboost as xgb
    pos = len(train_merge_3[train_merge_3['label']==1] ) * 1.0
    neg = len(train_merge_3[train_merge_3['label']==0] )
    clf = xgb.XGBClassifier(
        max_depth=6, n_estimators=1000, objective='binary:logistic',
        subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
        scale_pos_weight = neg/pos
    )
    X_train = np.array(train_merge_3[fea].as_matrix())
    y_train = np.array(train_merge_3['label'].as_matrix())
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='logloss', verbose=True)

    prediction = clf.predict_proba( np.array(test_merge_3[fea]) ).clip(0, 1)
    prediction = DataFrame(prediction)
    submit = DataFrame()
    submit['instanceID'] = test_merge_3['instanceID']
    submit['prob'] = prediction[1]
    submit = submit.sort_values(by='instanceID')
    submit.to_csv('../data/result/first.csv', index=False)


'''
函数说明：用训练集label列的均值，测下分数，再把均值当做测试集的提交答案提交下
'''
def second_model():
    import scipy as sp
    def logloss(act, pred):
        epsilon = 1e-15
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1 - epsilon, pred)
        ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
        ll = ll * -1.0 / len(act)
        return ll

    train_merge_3 = pd.read_csv('../data/merge_data/train_merge_3.csv')
    test_merge_3 = pd.read_csv('../data/merge_data/test_merge_3.csv')
    p = train_merge_3['label'].mean()  # Our predicted probability
    print('Predicted score:', logloss(train_merge_3['label'], np.zeros_like(train_merge_3['label']) + p))
    # ('Predicted score:', 0.11644123567464253)
    sub = pd.DataFrame({'instanceID': test_merge_3['instanceID'], 'prob': p})
    sub = sub.sort_values(by='instanceID')
    sub.to_csv('../data/result/second.csv', index=False)
    print sub.head()

'''
函数说明：为了测出来预测集中的正负比例，做份全0 1的结果出来
'''
def third_model():
    test_merge_3 = pd.read_csv('../data/merge_data/test_merge_3.csv')
    test_merge_3 = test_merge_3[['instanceID','label']]
    test_merge_3['label'] = 0
    test_merge_3 = test_merge_3.sort_values(by='instanceID')
    test_merge_3.to_csv('../data/result/third_0.csv', index=False)
    test_merge_3['label'] = 1
    test_merge_3 = test_merge_3.sort_values(by='instanceID')
    test_merge_3.to_csv('../data/result/third_1.csv', index=False)


'''
这种参数 对xgb.XGBClassifier 肯定过拟合了！但是这个参数对 xgb.train 却是好的！
函数说明：这是一个xgb模型(分类型的)，不针对具体的数据集，用的时候值需要传进来参数即可！
参数说明：train->训练集数据 test->预测集数据 fea->特征列
'''
def model_xgb_1(train, test, fea):
    print('model_xgb_1 function')
    import xgboost as xgb
    pos = len(train[train['label'] == 1]) * 1.0
    neg = len(train[train['label'] == 0])
    clf = xgb.XGBClassifier(
        max_depth=6, n_estimators=1000, objective='binary:logistic',
        subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
        scale_pos_weight=neg / pos,learning_rate=0.01,min_child_weight=5
    )
    X_train = np.array(train[fea].as_matrix())
    y_train = np.array(train['label'].as_matrix())
    print('begin to make model')
    # clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='logloss', verbose=True)
    clf.fit(X_train, y_train)
    print('make over')
    prediction = clf.predict_proba(np.array(test[fea])).clip(0, 1)
    prediction = DataFrame(prediction)
    submit = DataFrame()
    submit['instanceID'] = test['instanceID']
    submit['prob'] = prediction[1]
    submit = submit.sort_values(by='instanceID')
    submit.to_csv('../data/result/xgb_classi.csv', index=False)


'''
这种参数 对xgb.XGBClassifier 肯定过拟合了！但是这个参数对 xgb.train 却是好的！
函数说明：这是一个xgb模型(分类型的)，不针对具体的数据集，用的时候值需要传进来参数即可！
参数说明：train->训练集数据 test->预测集数据 fea->特征列
'''
def ceate_feature_map(features):
    outfile = open('../data/result/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
def model_xgb_1_1(train, test, fea):
    print('model_xgb_1_1 function')
    print('xgb_14_3500.csv')
    import xgboost as xgb
    import operator
    clf = xgb.XGBClassifier(
        learning_rate=0.01, n_estimators=3500, max_depth=6,
        min_child_weight=4, gamma=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1,
        objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=27)

    X_train = np.array(train[fea].as_matrix())
    y_train = np.array(train['label'].as_matrix())
    print('begin to make model')

    clf.fit(X_train, y_train)

    # 打印特征重要性的代码
    ceate_feature_map(fea)
    importance = clf.booster().get_fscore(fmap='../data/result/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    df.to_csv("../data/result/featureImportance_14_3500.csv", index=False)


    print('make over')
    prediction = clf.predict_proba(np.array(test[fea])).clip(0, 1)
    prediction = DataFrame(prediction)
    submit = DataFrame()
    submit['instanceID'] = test['instanceID']
    submit['prob'] = prediction[1]
    submit = submit.sort_values(by='instanceID')
    submit.to_csv('../data/result/xgb_14_3500.csv', index=False)



'''
函数说明：这是一个xgb模型(xgb.train型的)，不针对具体的数据集，用的时候值需要传进来参数即可！
参数说明：train->训练集数据 test->预测集数据 fea->特征列
'''
def model_xgb_2(train, test, fea):
    import xgboost as xgb
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'gamma' : 0.1,
              'learning_rate': 0.01,
              'eval_metric': 'logloss',
              'min_child_weight': 4,
              'max_depth': 6,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.8,
              'tree_method': 'exact',
              'seed': 0,
              'silent': 1,
              'lambda ':1,
              'alpha':0,
              'scale_pos_weight':1
              }

    print('begin make model')
    dtrain = xgb.DMatrix(train[fea].as_matrix(), train['label'].as_matrix())  # 训练集的所有特征列，训练集的“要预测的那一列
    bst = xgb.train(params, dtrain, num_boost_round=5000)
    print('make model over')
    prediction = bst.predict(xgb.DMatrix(test[fea].as_matrix())).clip(0, 1)

    submit = DataFrame()

    submit['instanceID'] = test['instanceID']
    submit['prob'] = prediction
    submit = submit.sort_values(by='instanceID')
    submit.to_csv('../data/result/xgb_train.csv', index=False)




'''
函数说明：xgboost做kf来训练预测结果
参数说明：train->训练集数据 test->预测集数据 fea->特征列
'''
def xgboost_model_kf(train,test,fea):
    print('move to light_gbm model')
    import xgboost as xgb
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'gamma': 0.1,
              'learning_rate': 0.01,
              'eval_metric': 'logloss',
              'min_child_weight': 4,
              'max_depth': 6,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.8,
              'tree_method': 'exact',
              'seed': 0,
              'silent': 1,
              'lambda ': 1,
              'alpha': 0,
              'scale_pos_weight': 1
              }
    # 训练集（特征列）
    col = fea
    train_data = train[col]
    train_data = np.array(train_data)
    # 训练集（label列）
    train_target = train['label']
    train_target = np.array(train_target)

    # 生成2个“list”变量，1个list变量是0 ~ len(train_data)中的1259个数字(test_index)
    # 另一个list变量是其中的 2518 个数字(train_index)，而且两个list互不相重复
    num_fold = 0
    random_state = 51
    models = []
    nfolds = 10
    kf = KFold(len(train_data), n_folds=nfolds, shuffle=True, random_state=random_state)
    for train_index, test_index in kf:  # train_index test_index 在 5次循环中每一次都不同，但是每一次的这两个list合起来刚好是 0~len(train_data)中的每一个数字
        print(u'which model begin to make : ', num_fold)
        num_fold += 1

        X_train = train_data[train_index]
        X_train = DataFrame(X_train)  # 训练集的“特征列”（DataFrame类型）
        Y_train = train_target[train_index]
        Y_train = DataFrame(Y_train)
        Y_train = Y_train[0]  # 训练集的“label列”（Series类型）

        X_valid = train_data[test_index]
        X_valid = DataFrame(X_valid)  # 测试集的“特征列”（DataFrame类型）
        Y_valid = train_target[test_index]
        Y_train = DataFrame(Y_train)
        Y_train = Y_train[0]  # 测试集的“label列”（DataFrame类型）

        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        print('begin to make model')
        dtrain = xgb.DMatrix(X_train.as_matrix(), Y_train.as_matrix())  # 训练集的所有特征列，训练集的“要预测的那一列
        bst = xgb.train(params, dtrain, num_boost_round=5000)
        print('make model over')

        models.append(bst)  # 将上面刚刚训练好的模型存储起来

    # 用刚刚生成的 5 个模型，预测出来5份结果，然后取均值

    num_fold = 0
    yfull_test = []
    for j in range(nfolds):
        print(u'which model we will use to predicte : ', num_fold)
        model = models[j]  # 先拿第一个模型来搞
        num_fold += 1
        y_pred = model.predict(xgb.DMatrix(test[col].as_matrix())).clip(0, 1)
        yfull_test.append(list(y_pred))

    a = np.array(yfull_test[0])  # 第一个模型的预测结果
    for j in range(1, nfolds):
        a += np.array(yfull_test[j])
    a /= nfolds

    submit = DataFrame()

    submit['instanceID'] = test['instanceID']
    submit['prob'] = a
    submit = submit.sort_values(by='instanceID')
    submit.to_csv('../data/result/xgb_train_kf.csv', index=False)








'''
函数说明：这是一个gbdt模型，不针对具体的数据集，用的时候值需要传进来参数即可！
参数说明：train->训练集数据 test->预测集数据 fea->特征列
'''
def gbdt_model(train,test,fea):
    print('gbdt_model function')
    from sklearn.ensemble import GradientBoostingClassifier
    original_params = {'n_estimators': 5000,
                       'max_depth': 6,
                       'learning_rate':0.01 ,
                       'loss': 'deviance',

                       }
    params = dict(original_params)
    gbdt = GradientBoostingClassifier(**params)

    # 预测
    print('begin to build model')
    gbdt.fit(train[fea].as_matrix(), train['label'].as_matrix())  # 训练出来模型
    print('build model over')

    prediction = gbdt.predict_proba(test[fea].as_matrix())
    prediction = DataFrame(prediction)

    submit = DataFrame()
    submit['instanceID'] = test['instanceID']
    print submit.head()
    submit['prob'] = prediction[1]
    submit = submit.sort_values(by='instanceID')
    submit.to_csv('../data/result/gbdt.csv', index=False)

'''
函数说明：这是一个"随机森林"模型，不针对具体的数据集，用的时候值需要传进来参数即可！
参数说明：train->训练集数据 test->预测集数据 fea->特征列
'''
def random_forest_model(train,test,fea):
    print('random_forest_model function')

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=500, max_features=7,n_jobs=-1)  # 单棵树最大深度采用默认,n_jobs=-1 代表使用所有的核数！

    # 预测
    print('begin to build model')
    clf.fit(np.array(train[fea]), np.array(train['label']))
    print('build model over')

    prediction = clf.predict_proba(np.array(test[fea]))
    prediction = DataFrame(prediction)

    submit = DataFrame()
    submit['instanceID'] = test['instanceID']

    submit['prob'] = prediction[1]
    submit = submit.sort_values(by='instanceID')
    submit.to_csv('../data/result/random_fore.csv', index=False)


'''
函数说明：这是一个"lightgbm"模型，不针对具体的数据集，用的时候值需要传进来参数即可！
参数说明：train->训练集数据 test->预测集数据 fea->特征列
'''
def light_gbm_model(train,test,fea):
    print('light_gbm_model function')

    print('move to light_gbm model')
    import lightgbm as lgb
    from sklearn.cross_validation import KFold
    # 训练集（特征列）
    col = fea
    train_data = train[col]
    train_data = np.array(train_data)
    # 训练集（label列）
    train_target = train['label']
    train_target = np.array(train_target)

    # 生成2个“list”变量，1个list变量是0 ~ len(train_data)中的1259个数字(test_index)
    # 另一个list变量是其中的 2518 个数字(train_index)，而且两个list互不相重复
    num_fold = 0
    random_state = 51
    models = []
    nfolds = 5
    kf = KFold(len(train_data), n_folds=nfolds, shuffle=True, random_state=random_state)
    for train_index, test_index in kf:  # train_index test_index 在 5次循环中每一次都不同，但是每一次的这两个list合起来刚好是 0~len(train_data)中的每一个数字
        print(u'which model begin to make : ', num_fold)
        num_fold += 1
        model = lgb.LGBMRegressor(
            boosting_type="gbdt",
            max_depth=6,
            learning_rate=0.01,
            objective='binary',
            min_child_weight=4,
            colsample_bytree=0.8,
            n_estimators=3500)
        X_train = train_data[train_index]
        X_train = DataFrame(X_train)  # 训练集的“特征列”（DataFrame类型）
        Y_train = train_target[train_index]
        Y_train = DataFrame(Y_train)
        Y_train = Y_train[0]  # 训练集的“label列”（Series类型）

        X_valid = train_data[test_index]
        X_valid = DataFrame(X_valid)  # 测试集的“特征列”（DataFrame类型）
        Y_valid = train_target[test_index]
        Y_train = DataFrame(Y_train)
        Y_train = Y_train[0]  # 测试集的“label列”（DataFrame类型）

        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        print('begin to make model')
        model.fit(X_train, Y_train,
                  eval_set=[(X_valid, Y_valid)],
                  eval_metric='logloss',
                  )
        print('make model over')

        models.append(model)  # 将上面刚刚训练好的模型存储起来

    # 用刚刚生成的 5 个模型，预测出来5份结果，然后取均值

    num_fold = 0
    yfull_test = []
    for j in range(nfolds):
        print(u'which model we will use to predicte : ', num_fold)
        model = models[j]  # 先拿第一个模型来搞
        num_fold += 1
        y_pred = model.predict(test[col], num_iteration=model.best_iteration)
        # print(y_pred)
        yfull_test.append(list(y_pred))

    a = np.array(yfull_test[0])  # 第一个模型的预测结果
    for j in range(1, nfolds):
        a += np.array(yfull_test[j])
    a /= nfolds

    submit = DataFrame()

    submit['instanceID'] = test['instanceID']
    submit['prob'] = a
    submit = submit.sort_values(by='instanceID')
    submit.to_csv('../data/result/light_gbm.csv', index=False)


'''
函数说明：将几个好的结果加权融合下！
'''
def ronghe():
    light_gbm = pd.read_csv('../data/result/light_gbm.csv')
    xgb_14_3500 = pd.read_csv('../data/result/xgb_14_3500.csv')

    light_gbm = light_gbm.sort_values(by='instanceID')
    xgb_14_3500 = xgb_14_3500.sort_values(by='instanceID')

    print light_gbm.head()
    print xgb_14_3500.head()

    light_gbm['prob'] = light_gbm['prob']*0.5 + xgb_14_3500['prob']*0.5

    light_gbm.to_csv('../data/result/l_x.csv', index=False, mode='a', encoding='utf-8')



'''
函数说明：这块尝试出来最合适的一个线下@xgb.train。这块选择的是 等比例分配
'''
import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll
def fun_1():

    train_merge_3 = pd.read_csv('../data/merge_data/train_merge_8.csv')
    del train_merge_3['conversionTime']
    fea = list(train_merge_3.columns)
    fea.remove('label')
    fea.remove('clickTime_day')

    train_pos = train_merge_3[train_merge_3['label'] == 1]
    train_neg = train_merge_3[train_merge_3['label'] == 0]
    # 将正样本按 4:1 分成两部分
    from sklearn import cross_validation
    train_pos_4, train_pos_1 = cross_validation.train_test_split(train_pos, test_size=0.2, random_state=0)
    # 负样本
    train_neg_4, train_neg_1 = cross_validation.train_test_split(train_neg, test_size=0.2, random_state=0)

    train = pd.concat([train_pos_4, train_neg_4], axis=0)
    test = pd.concat([train_pos_1, train_neg_1], axis=0)

    # 线如今的 train_4就是训练集、train_1就是测试集
    print('model_xgb_2 function')
    import xgboost as xgb
    params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'learning_rate': 0.01,
                  'eval_metric': 'logloss',
                  'min_child_weight': 5,
                  'max_depth': 6,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'colsample_bylevel': 0.8,
                  'tree_method': 'exact',
                  'seed': 0,
                  'silent': 1,
                  'scale_pos_weight':1
                  }
    print('begin make model')
    dtrain = xgb.DMatrix(train[fea].as_matrix(), train['label'].as_matrix())
    dtest = xgb.DMatrix(test[fea].as_matrix(), test['label'].as_matrix())
    watchlist = [(dtrain,'train'),(dtest,'test')]

    bst = xgb.train(params, dtrain, num_boost_round=1000,evals=watchlist )
    # bst = xgb.cv(params, dtrain, num_boost_round=1000 , nfold=5 , metrics ='logloss',verbose_eval=True )

    print('make model over')
    prediction = bst.predict(xgb.DMatrix(test[fea].as_matrix())).clip(0, 1)
    print('Predicted score:', logloss(test['label'], prediction) )



'''
函数说明：使用xgb.cv这个函数，来做下线下尝试
'''
def fun_2():

    train_merge_3 = pd.read_csv('../data/merge_data/train_merge_5.csv')
    del train_merge_3['conversionTime']
    fea = list(train_merge_3.columns)
    fea.remove('label')
    fea.remove('clickTime_day')

    train_pos = train_merge_3[train_merge_3['label'] == 1]
    train_neg = train_merge_3[train_merge_3['label'] == 0]
    # 将正样本按 4:1 分成两部分
    from sklearn import cross_validation
    train_pos_4, train_pos_1 = cross_validation.train_test_split(train_pos, test_size=0.2, random_state=0)
    # 负样本
    train_neg_4, train_neg_1 = cross_validation.train_test_split(train_neg, test_size=0.2, random_state=0)

    train = pd.concat([train_pos_4, train_neg_4], axis=0)
    test = pd.concat([train_pos_1, train_neg_1], axis=0)

    # 线如今的 train_4就是训练集、train_1就是测试集
    print('model_xgb_2 function')
    import xgboost as xgb
    clf = xgb.XGBClassifier(
            max_depth=6, n_estimators=100, objective='binary:logistic',
            subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
            scale_pos_weight=40
        )
    print('begin make model')
    X_train = np.array(train[fea].as_matrix())
    y_train = np.array(train['label'].as_matrix())


    clf.fit(X_train, y_train)
    print('make over')
    prediction = clf.predict_proba(np.array(test[fea])).clip(0, 1)
    prediction = DataFrame(prediction)
    print('Predicted score:', logloss(test['label'], prediction[1]) )


'''
函数说明：线下测试！
'''
def off_line():

    print('mv to off_line function')

    train_merge_9 = pd.read_csv('../data/merge_data/train_merge_9.csv')

    test = train_merge_9[train_merge_9['clickTime_day'] == 29]

    train_merge_9 = train_merge_9[train_merge_9['clickTime_day'] != 30]
    train_merge_9 = train_merge_9[train_merge_9['clickTime_day'] != 29]

    '''看看要删去什么特征'''
    # 初始化4维特征
    # 历史点击量：x + y -》i_lishi_dianji
    # 历史转化量：y     -> i_lishi_zhuanhualiang
    # 转化率: y/(x+y)      -》i_lishi_bilv
    # 点击量占比：（x + y） / (x + y)那一列求和 ->i_dianji_bi
    # 转化量占比： y / y那一列求和 -> i_zhuanhua_bi
    # 个人转化率 刻画 用户 性格： y / x -> i_geren_zhaunhua

    train = train_merge_9.drop(train_merge_9.filter(regex='lishi_dianji|lishi_zhuanhualiang|dianji_bi|zhuanhua_bi|geren_zhaunhua'), axis=1)
    test = test.drop(test.filter(regex='lishi_dianji|lishi_zhuanhualiang|dianji_bi|zhuanhua_bi|geren_zhaunhua'), axis=1)

    fea = list(train.columns)
    fea.remove('label')
    fea.remove('clickTime_day')
    fea.remove('conversionTime')  # 特征列

    import xgboost as xgb
    import operator
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'learning_rate': 0.01,
              'eval_metric': 'logloss',
              'min_child_weight': 4,
              'max_depth': 6,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.8,
              'tree_method': 'exact',
              'silent': 1,
              'scale_pos_weight': 1,
              'gamma':0.1,
              'nthread':8,
              'lambda':1,
              'alpha':0
              }
    print('begin to make model')
    dtrain = xgb.DMatrix(train[fea].as_matrix(), train['label'].as_matrix())
    dtest = xgb.DMatrix(test[fea].as_matrix(), test['label'].as_matrix())
    watchlist = [(dtrain, 'train'), (dtest, 'test')]

    bst = xgb.train(params, dtrain, num_boost_round=5000, evals=watchlist)





if __name__ == "__main__":

    train_merge_10 = pd.read_csv('../data/merge_data/train_merge_14.csv')
    test_merge_10 = pd.read_csv('../data/merge_data/test_merge_14.csv')

    # # train = train_merge_9.drop(
    # #     train_merge_9.filter(regex='_lishi_bilv|lishi_dianji|lishi_zhuanhualiang|dianji_bi|zhuanhua_bi|geren_zhaunhua'), axis=1)
    # # >> > len(train.columns)
    # # 50  基础特征只剩下50维了！（还包括lable那些不是特征的列）
    # # test = test_merge_9.drop(test_merge_9.filter(regex='_lishi_bilv|lishi_dianji|lishi_zhuanhualiang|dianji_bi|zhuanhua_bi|geren_zhaunhua'), axis=1)
    # print('len(train_merge_9.columns):',len(train_merge_9.columns))
    # print('len(test_merge_9.columns):', len(test_merge_9.columns))
    # train = train_merge_9.drop(
    #     train_merge_9.filter(regex='_lishi_bilv|lishi_dianji|lishi_zhuanhualiang|dianji_bi|zhuanhua_bi'),
    #     axis=1)
    # test = test_merge_9.drop(
    #     test_merge_9.filter(regex='_lishi_bilv|lishi_dianji|lishi_zhuanhualiang|dianji_bi|zhuanhua_bi'),
    #     axis=1)
    # print('len(train.columns):', len(train.columns))
    # print('len(test.columns):', len(test.columns))

    # print(len(train_merge_9))
    # train = train_merge_9[train_merge_9['clickTime_day'] != 30]
    #
    # train_will_append = train_merge_9[train_merge_9['clickTime_day'] == 30]
    # train_will_append = train_will_append[train_will_append['label'] == 1]
    # train = train.append(train_will_append)


    # print(len(train))

    # lie = ['count_cate_1_-1', 'count_cate_1_1', 'count_cate_1_2', 'count_cate_1_3', 'count_cate_1_4', 'count_cate_1_5', 'count_cate_2_-1',
    #  'count_cate_2_1', 'count_cate_2_2', 'count_cate_2_3', 'count_cate_2_4', 'count_cate_2_5', 'count_cate_2_6', 'count_cate_2_7', 'count_cate_2_8',
    # 'count_cate_2_9', 'count_cate_2_10', 'count_cate_2_11']
    # for i in lie:
    #     del train_merge_4[i]
    #     del test_merge_4[i]

    # lie = ['count_click']
    # for i in lie:
    #     del train_merge_4[i]
    #     del test_merge_4[i]



    '''----------------------------------------------------------------------------------------------------------------------------'''
    '''删除 appCategory 的一些特征，实验 appCategory_1 的特征'''
    print('len(train_merge_10):', len(train_merge_10.columns))
    print('len(test_merge_10):', len(test_merge_10.columns))
    list_6 = [201, 409, 301, 203, 503, 407, 0, 103, 406, 209, 108, 211, 402, 210, 2, 405, 408, 106, 403, 401, 109, 104,
              303, 110, 105, 204, 1]
    for i in range(len(list_6)):
        list_6[i] = 'appCategory' + str(list_6[i])
    train_merge_10 = train_merge_10.drop(train_merge_10[list_6], axis=1)
    test_merge_10 = test_merge_10.drop(test_merge_10[list_6], axis=1)
    print('len(train_merge_10):', len(train_merge_10.columns))
    print('len(test_merge_10):', len(test_merge_10.columns))

    '''添加上比例试下'''
    # list_6 = [2, 4, 3, 5, -1, 1]
    # for i in range(len(list_6)):
    #     list_6[i] = 'appCategory_1' + str(list_6[i])
    # train['appCategory_1_sum'] = 0
    # test['appCategory_1_sum'] = 0
    # for i in list_6:
    #     train['appCategory_1_sum'] += train[i]
    #     test['appCategory_1_sum'] += test[i]
    # for i in list_6:
    #     train[i + '_rate'] = train[i] / train['appCategory_1_sum']
    #     test[i + '_rate'] = test[i] / test['appCategory_1_sum']
    # print('len(train):', len(train.columns))
    # print('len(test):', len(test.columns))
    # # 再删除原先的 数值统计 特征
    # train = train.drop(train[list_6], axis=1)
    # test = test.drop(test[list_6], axis=1)
    # del train['appCategory_1_sum']
    # del test['appCategory_1_sum']
    # print('len(train):', len(train.columns))
    # print('len(test):', len(test.columns))



    '''删除 appCategory_1 的一些特征，实验 appCategory 的特征'''
    list_6 = [2, 4, 3, 5, -1, 1]
    for i in range(len(list_6)):
        list_6[i] = 'appCategory_1' + str(list_6[i])

    train_merge_10 = train_merge_10.drop(train_merge_10[list_6], axis=1)
    test_merge_10 = test_merge_10.drop(test_merge_10[list_6], axis=1)

    print('len(train_merge_10):', len(train_merge_10.columns))
    print('len(test_merge_10):', len(test_merge_10.columns))

    '''添加上比例试下'''
    # list_6 = [201, 409, 301, 203, 503, 407, 0, 103, 406, 209, 108, 211, 402, 210, 2, 405, 408, 106, 403, 401, 109, 104,
    #           303, 110, 105, 204, 1]
    # for i in range(len(list_6)):
    #     list_6[i] = 'appCategory' + str(list_6[i])
    # train['appCategory_sum'] = 0
    # test['appCategory_sum'] = 0
    # for i in list_6:
    #     train['appCategory_sum'] += train[i]
    #     test['appCategory_sum'] += test[i]
    # for i in list_6:
    #     train[i + '_rate'] = train[i] / train['appCategory_sum']
    #     test[i + '_rate'] = test[i] / test['appCategory_sum']
    # print('len(train):', len(train.columns))
    # print('len(test):', len(test.columns))
    # # 再删除原先的 数值统计 特征
    # train = train.drop(train[list_6], axis=1)
    # test = test.drop(test[list_6], axis=1)
    # del train['appCategory_sum']
    # del test['appCategory_sum']
    # print('len(train):', len(train.columns))
    # print('len(test):', len(test.columns))




    '''$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'''
    '''删除 appCategory 的一些特征，实验 appCategory_1 的特征'''
    # print('len(train_merge_10):', len(train_merge_10.columns))
    # print('len(test_merge_10):', len(test_merge_10.columns))
    # list_6 = [201, 409, 301, 203, 503, 407, 0, 103, 406, 209, 108, 211, 402, 210, 2, 405, 408, 106, 403, 401, 109, 104,
    #           303, 110, 105, 204, 1]
    # for i in range(len(list_6)):
    #     list_6[i] = 'user_actions_appCategory_' + str(list_6[i])
    # train_merge_10 = train_merge_10.drop(train_merge_10[list_6], axis=1)
    # test_merge_10 = test_merge_10.drop(test_merge_10[list_6], axis=1)



    '''添加上比例试下'''
    list_6 = [2, 4, 3, 5, -1, 1]
    for i in range(len(list_6)):
        list_6[i] = 'user_actions_appCategory_1_' + str(list_6[i])
    train_merge_10['appCategory_1_sum'] = 0
    test_merge_10['appCategory_1_sum'] = 0
    for i in list_6:
        train_merge_10['appCategory_1_sum'] += train_merge_10[i]
        test_merge_10['appCategory_1_sum'] += test_merge_10[i]
    for i in list_6:
        train_merge_10[i + '_rate'] = train_merge_10[i] / train_merge_10['appCategory_1_sum']
        test_merge_10[i + '_rate'] = test_merge_10[i] / test_merge_10['appCategory_1_sum']

    # 再删除原先的 数值统计 特征
    train_merge_10 = train_merge_10.drop(train_merge_10[list_6], axis=1)
    test_merge_10 = test_merge_10.drop(test_merge_10[list_6], axis=1)
    del train_merge_10['appCategory_1_sum']
    del test_merge_10['appCategory_1_sum']

    print('len(train_merge_10):', len(train_merge_10.columns))
    print('len(test_merge_10):', len(test_merge_10.columns))



    '''删除 appCategory_1 的一些特征，实验 appCategory 的特征'''
    # print('len(train_merge_10):', len(train_merge_10.columns))
    # print('len(test_merge_10):', len(test_merge_10.columns))
    # list_6 = [2, 4, 3, 5, -1, 1]
    # for i in range(len(list_6)):
    #     list_6[i] = 'user_actions_appCategory_1_' + str(list_6[i])
    #
    # train_merge_10 = train_merge_10.drop(train_merge_10[list_6], axis=1)
    # test_merge_10 = test_merge_10.drop(test_merge_10[list_6], axis=1)
    #
    # print('len(train_merge_10):', len(train_merge_10.columns))
    # print('len(test_merge_10):', len(test_merge_10.columns))

    '''添加上比例试下'''
    list_6 = [201, 409, 301, 203, 503, 407, 0, 103, 406, 209, 108, 211, 402, 210, 2, 405, 408, 106, 403, 401, 109, 104,
              303, 110, 105, 204, 1]
    for i in range(len(list_6)):
        list_6[i] = 'user_actions_appCategory_' + str(list_6[i])
    train_merge_10['appCategory_sum'] = 0
    test_merge_10['appCategory_sum'] = 0
    for i in list_6:
        train_merge_10['appCategory_sum'] += train_merge_10[i]
        test_merge_10['appCategory_sum'] += test_merge_10[i]
    for i in list_6:
        train_merge_10[i + '_rate'] = train_merge_10[i] / train_merge_10['appCategory_sum']
        test_merge_10[i + '_rate'] = test_merge_10[i] / test_merge_10['appCategory_sum']

    # 再删除原先的 数值统计 特征
    train_merge_10 = train_merge_10.drop(train_merge_10[list_6], axis=1)
    test_merge_10 = test_merge_10.drop(test_merge_10[list_6], axis=1)
    del train_merge_10['appCategory_sum']
    del test_merge_10['appCategory_sum']

    print('len(train_merge_10):', len(train_merge_10.columns))
    print('len(test_merge_10):', len(test_merge_10.columns))





    train = train_merge_10
    test = test_merge_10


    fea = list(train.columns)
    fea.remove('label')
    fea.remove('conversionTime')  # 特征列

    fea.remove('clickTime_day')
    fea.remove('clickTime')
    fea.remove('instanceID')
    del train_merge_10
    del test_merge_10

    # fea.remove('trick_userID_cha_time')


    # model_xgb_1_1(train, test, fea)
    light_gbm_model(train, test, fea)





