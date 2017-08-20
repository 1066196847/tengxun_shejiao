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
import xgboost as xgb
from xgboost import XGBClassifier
import time

'''
函数说明：这个函数用于来测试一套具体参数的，效果咋样
参数说明：alg->模型变量；dtrain->训练集；predictors->特征列名（所有）
'''
import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)

    return ll

def modelfit(alg, train, test, fea, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        dtrain = xgb.DMatrix(train[fea].values, label=train['label'].values)
        print('cv')
        cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='logloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0]) # 将“迭代次数”这个参数，重置成做完 cv 后最合适的“迭代次数”

    # Fit the algorithm on the data
    print('begin to make model')
    alg.fit(train[fea], train['label'], eval_metric='logloss')
    print('Will train until cv error hasn\'t decreased in ' + str(early_stopping_rounds) + ' rounds.')
    print('Stopping. Best iteration:')
    print(cvresult.tail(1))  # cvresult变量里面存储了每一次迭代时候的东西
    # 预测出来结果
    dtrain_predprob = alg.predict_proba(train[fea])[:, 1]
    dtest_predprob = alg.predict_proba(test[fea])[:, 1]

    # 打印“训练集”logloss
    print "train logloss :" % logloss(train['label'], dtrain_predprob)
    print "test logloss :" % logloss(test['label'], dtest_predprob)


'''
函数说明：第一次调节 迭代数量
'''
def a1():
    print('begin to prepare data')
    train_merge_4 = pd.read_csv('../data/merge_data/train_merge_4.csv')
    del train_merge_4['conversionTime']
    fea = list(train_merge_4.columns)
    fea.remove('label')

    train_pos = train_merge_4[train_merge_4['label'] == 1]
    train_neg = train_merge_4[train_merge_4['label'] == 0]
    # 将正样本按 4:1 分成两部分
    from sklearn import cross_validation
    train_pos_4, train_pos_1 = cross_validation.train_test_split(train_pos, test_size=0.2, random_state=0)
    # 负样本
    train_neg_4, train_neg_1 = cross_validation.train_test_split(train_neg, test_size=0.2, random_state=0)

    train = pd.concat([train_pos_4, train_neg_4], axis=0)
    test = pd.concat([train_pos_1, train_neg_1], axis=0)

    fea = [x for x in train.columns if x not in ['label', 'conversionTime']]  # 特征列
    print('prepare data over')

    xgb1 = xgb.XGBClassifier(
     learning_rate =0.1,
     n_estimators=3500,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=8,
     scale_pos_weight=1,
     seed=27)

    modelfit(xgb1, train,test, fea)

    # 结果
    # Will train until cv error hasn't decreased in 50 rounds.
    # Stopping. Best iteration:
    #      test-logloss-mean  test-logloss-std  train-logloss-mean  \
    # 841           0.101761          0.000481            0.097632
    #
    #      train-logloss-std
    # 841           0.000111
    # train logloss :
    # test logloss :





'''调节max_depth min_child_weight'''
def a2():
    '''不能使用这种方法，因为没有一个合适的评估方法'''
    # print('begin to prepare a2 data')
    # train = pd.read_csv('../data/merge_data/train_merge_4.csv')
    # fea = [x for x in train.columns if x not in ['label', 'conversionTime']]  # 特征列
    # print('prepare data over')
    #
    # from sklearn.grid_search import GridSearchCV
    # param_test1 = {
    #  'max_depth':range(3,10,2),
    #  'min_child_weight':range(1,6,2)
    # }
    #
    # gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=841, max_depth=5,
    #  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=8, scale_pos_weight=1, seed=27),
    #  param_grid = param_test1, scoring='neg_log_loss',n_jobs=8,iid=False, cv=5)
    # # n_jobs：多少个进程
    # # iid：默认是True，数据默认被看做“相同分布”，在cv时候会应用此默认情况，损失最小化是每个样品的总损失，而不是横跨褶皱的平均损失。
    # gsearch1.fit(train[fea], train['label'])
    # print(gsearch1.grid_scores_)
    # print(gsearch1.best_params_)
    # print(gsearch1.best_score_)
    #
    # # max_depth:3, min_child_weight:3
    # # -0.4651164


    '''使用下面这种for循环来做吧'''
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
    X_train = np.array(train[fea].as_matrix())
    y_train = np.array(train['label'].as_matrix())


    # 线如今的 train就是训练集、test就是测试集

    import xgboost as xgb

    max_depth_list = [3,5,7,9]
    min_child_weight_list = [1,3,5,7]
    for i in range(0,4):
        for j in range(0,4):
            clf = xgb.XGBClassifier(
                learning_rate=0.1, n_estimators=841, max_depth=max_depth_list[i],
                min_child_weight=min_child_weight_list[j], gamma=0, subsample=0.8, colsample_bytree=0.8,
                objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=27)

            clf.fit(X_train, y_train)

            prediction = clf.predict_proba(np.array(test[fea])).clip(0, 1)
            prediction = DataFrame(prediction)
            print 'max_depth: ',max_depth_list[i],'min_child_weight: ',min_child_weight_list[j],'score: ',logloss(test['label'], prediction[1])
    # 从搜索结果看来，max_depth取5 min_child_weight取3的时候，测试集分数最佳


'''
函数说明：在上面的粗略结果中，再次细调出最好的参数
'''
def a3():
    '''使用下面这种for循环来做吧'''
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
    X_train = np.array(train[fea].as_matrix())
    y_train = np.array(train['label'].as_matrix())

    # 线如今的 train就是训练集、test就是测试集

    import xgboost as xgb
    max_depth_list = [4,5,6]
    min_child_weight_list = [2,3,4]
    for i in range(0, 3):
        for j in range(0, 3):
            clf = xgb.XGBClassifier(
                learning_rate=0.1, n_estimators=841, max_depth=max_depth_list[i],
                min_child_weight=min_child_weight_list[j], gamma=0, subsample=0.8, colsample_bytree=0.8,
                objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=27)

            clf.fit(X_train, y_train)

            prediction = clf.predict_proba(np.array(test[fea])).clip(0, 1)
            prediction = DataFrame(prediction)
            print 'max_depth: ', max_depth_list[i], 'min_child_weight: ', min_child_weight_list[j], 'score: ', logloss(test['label'], prediction[1])
    # 最终决定 max_depth:6 min_child_weight:4 这两个参数是最优的参数，对应测试集分数是：0.10136


'''调节 gamma 参数：叶子节点继续分裂的时候，所需要的最小损失'''
def a4():
    '''使用下面这种for循环来做吧'''
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
    X_train = np.array(train[fea].as_matrix())
    y_train = np.array(train['label'].as_matrix())

    # 线如今的 train就是训练集、test就是测试集

    import xgboost as xgb
    gamma_list = [0,0.1,0.2,0.3,0.4,0.5]
    for i in range(0, len(gamma_list)):
        clf = xgb.XGBClassifier(
            learning_rate=0.1, n_estimators=841, max_depth=6,
            min_child_weight=4, gamma=gamma_list[i], subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=27)

        clf.fit(X_train, y_train)

        prediction = clf.predict_proba(np.array(test[fea])).clip(0, 1)
        prediction = DataFrame(prediction)
        print 'gamma: ', gamma_list[i], 'score: ', logloss(test['label'], prediction[1])
    # gamma参数最好是：0.1


def a5():
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

    fea = [x for x in train.columns if x not in ['label', 'conversionTime']]  # 特征列
    print('prepare data over')

    xgb2 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1500,
        max_depth=6,
        min_child_weight=4,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=27)
    modelfit(xgb2, train, test, fea)
    # 588次，测试集得分是 0.101429 训练集得分是 0.0961


'''
函数说明：调节 subsample colsample_bytree
'''
def a6():
    '''使用下面这种for循环来做吧'''
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
    X_train = np.array(train[fea].as_matrix())
    y_train = np.array(train['label'].as_matrix())

    # 线如今的 train就是训练集、test就是测试集

    import xgboost as xgb
    subsample_list = [0.6,0.7,0.8,0.9]
    colsample_bytree_list = [0.6,0.7,0.8,0.9]
    for i in range(0, 4):
        for j in range(0, 4):
            clf = xgb.XGBClassifier(
                learning_rate=0.1, n_estimators=588, max_depth=6,
                min_child_weight=4, gamma=0.1, subsample=subsample_list[i], colsample_bytree=colsample_bytree_list[j],
                objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=27)

            clf.fit(X_train, y_train)

            prediction = clf.predict_proba(np.array(test[fea])).clip(0, 1)
            prediction = DataFrame(prediction)
            print 'subsample: ', subsample_list[i], 'colsample_bytree: ', colsample_bytree_list[j], 'score: ', logloss(test['label'], prediction[1])
    # 两个参数都是 0.8，测试集分数是 0.101324


'''
函数说明：上面a6()之后，详细调节 subsample colsample_bytree 这两个参数
'''
def a7():
    '''使用下面这种for循环来做吧'''
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
    X_train = np.array(train[fea].as_matrix())
    y_train = np.array(train['label'].as_matrix())

    # 线如今的 train就是训练集、test就是测试集

    import xgboost as xgb
    subsample_list = [0.75,0.8,0.85]
    colsample_bytree_list = [0.75,0.8,0.85]
    for i in range(0, 3):
        for j in range(0, 3):
            clf = xgb.XGBClassifier(
                learning_rate=0.1, n_estimators=588, max_depth=6,
                min_child_weight=4, gamma=0.1, subsample=subsample_list[i], colsample_bytree=colsample_bytree_list[j],
                objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=27)

            clf.fit(X_train, y_train)

            prediction = clf.predict_proba(np.array(test[fea])).clip(0, 1)
            prediction = DataFrame(prediction)
            print 'subsample: ', subsample_list[i], 'colsample_bytree: ', colsample_bytree_list[j], 'score: ', logloss(test['label'], prediction[1])

    # 细调之后还是 0.8 0.8，分数是 0.1013244


'''调节 reg_alpha 参数：正则表达参数'''
def a8():
    '''使用下面这种for循环来做吧'''
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
    X_train = np.array(train[fea].as_matrix())
    y_train = np.array(train['label'].as_matrix())

    # 线如今的 train就是训练集、test就是测试集

    import xgboost as xgb
    reg_alpha_list = [0, 0.001, 0.005, 0.01, 0.05]
    for i in range(0, len(reg_alpha_list)):
        clf = xgb.XGBClassifier(
            learning_rate=0.1, n_estimators=588, max_depth=6,
            min_child_weight=4, gamma=0.1, subsample=0.8, colsample_bytree=0.8,reg_alpha= reg_alpha_list[i],
            objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=27)

        clf.fit(X_train, y_train)

        prediction = clf.predict_proba(np.array(test[fea])).clip(0, 1)
        prediction = DataFrame(prediction)
        print 'reg_alpha: ', reg_alpha_list[i], 'score: ', logloss(test['label'], prediction[1])
    # reg_alpha 选0最合适，分数是 0.1013244


'''降低学习速率，找出最合适的迭代数量'''
def a9():
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

    fea = [x for x in train.columns if x not in ['label', 'conversionTime']]  # 特征列
    print('prepare data over')

    xgb2 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=6,
        min_child_weight=4,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        reg_alpha=0,
        seed=27)
    modelfit(xgb2, train, test, fea)
    # 4999 0.101248


'''调节 scale_pos_weight 参数'''
def a10():
    '''使用下面这种for循环来做吧'''
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
    X_train = np.array(train[fea].as_matrix())
    y_train = np.array(train['label'].as_matrix())

    # 线如今的 train就是训练集、test就是测试集

    import xgboost as xgb
    scale_pos_weight_list = [0.9,0.8,0.7,0.6,0.4]
    for i in range(0, len(scale_pos_weight_list)):
        clf = xgb.XGBClassifier(
            learning_rate=0.01, n_estimators=1000, max_depth=6,
            min_child_weight=4, gamma=0.1, subsample=0.8, colsample_bytree=0.8,reg_alpha=0 ,
            objective='binary:logistic', nthread=8, scale_pos_weight=scale_pos_weight_list[i], seed=27)

        clf.fit(X_train, y_train)

        prediction = clf.predict_proba(np.array(test[fea])).clip(0, 1)
        prediction = DataFrame(prediction)
        print 'scale_pos_weight: ', scale_pos_weight_list[i], 'score: ', logloss(test['label'], prediction[1])

    # scale_pos_weight为 1 的时候 ：0.1028799
    # scale_pos_weight为 0.9 的时候 ：0.10302
    # scale_pos_weight为 0.8 时候 ：0.10343
    # scale_pos_weight为 0.5 时候 ：0.10760


'''调节 l1正则化那个参数，就是lambda'''
def a11():
    '''使用下面这种for循环来做吧'''
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
    X_train = np.array(train[fea].as_matrix())
    y_train = np.array(train['label'].as_matrix())

    # 线如今的 train就是训练集、test就是测试集

    import xgboost as xgb
    reg_lambda_list = [1,5,10,15,20]
    for i in range(0, len(reg_lambda_list)):
        clf = xgb.XGBClassifier(
            learning_rate=0.01, n_estimators=1000, max_depth=6,
            min_child_weight=4, gamma=0.1, subsample=0.8, colsample_bytree=0.8,reg_alpha=0 ,reg_lambda= reg_lambda_list[i],
            objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=27)

        clf.fit(X_train, y_train)

        prediction = clf.predict_proba(np.array(test[fea])).clip(0, 1)
        prediction = DataFrame(prediction)
        print 'reg_lambda_weight: ', reg_lambda_list[i], 'score: ', logloss(test['label'], prediction[1])




if __name__ == '__main__':
    '''第一次调节迭代数量'''
    # a1()
    '''第二次调节max_depth min_child_weight'''
    # a2()
    '''细调'''
    # a3()
    '''gamma参数'''
    # a4()
    '''再次调节下 迭代数量'''
    # a5()
    '''调节 subsample colsample_bytree'''
    # a6()
    '''详细调节下 subsample colsample_bytree'''
    # a7()
    '''调节下 reg_alpha 参数 '''
    # a8()
    '''降低学习速率，找出最合适的迭代数量'''
    # start_time = time.time()
    # a9()
    # print('cost time: {} seconds'.format(round(time.time() - start_time, 2)))
    '''scale_pos_weight'''
    # start_time = time.time()
    # a10()
    # print('cost time: {} seconds'.format(round(time.time() - start_time, 2)))
    '''还有Lambda这个参数也没调'''
    start_time = time.time()
    a11()
    print('cost time: {} seconds'.format(round(time.time() - start_time, 2)))



