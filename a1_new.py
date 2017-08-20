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
import string

'''
a1~a9 这些函数是将所有基本的 id特征提取在训练集、测试集中
其中 a1~a7 是做的一些数据分析，和数据基本处理，a9才是将所有的id特征连接的函数
'''
def a1():
    '''app_categories.csv'''
    # 1:查看下这个文件两列是否有空
    app_categories = pd.read_csv('../data/ago/app_categories.csv',dtype={'appID': str,'appCategory': 'int'})
    app_categories['appCategory_new'] = app_categories['appCategory']
    for i in app_categories.columns:
        print  i,app_categories[i].isnull().sum()
    # appID 0
    # appCategory 0

    # 2:接下来人工查看 app_categories 这个文件下第二列，发现大多数都是都是3个数字，但是还有一些是1个数字
    # 接着查看 FAQ 后发现，一个数字的表示：只有一级类目，没有二级类目（“210”表示一级类目编号为2，二级类目编号为10，类目未知或者无法获取时）

    # 首先是将那些“一位数字，但非0”的处理下，乘以100
    app_categories.loc[app_categories['appCategory_new'] < 100 , 'appCategory_new'] = app_categories.loc[app_categories['appCategory_new'] < 100 , 'appCategory_new'] * 100
    # 接下来就是那些“0”了
    app_categories['appCategory_new'] = app_categories['appCategory_new'].astype(str)
    app_categories['appCategory_new'] = app_categories['appCategory_new'].apply(lambda x: string.zfill(x, 3))  # 前向填充

    # 如“210”表示一级类目编号为2，二级类目编号为10，类目未知或者无法获取时，标记为0。
    # 所以将原先的 3个数字，分成两个特征，然后删除原先的这个
    app_categories['appCategory_1'] = app_categories['appCategory_new'].str[:1]
    app_categories['appCategory_2'] = app_categories['appCategory_new'].str[1:]
    # del app_categories['appCategory']
    for i in app_categories.columns:
        app_categories[i] = app_categories[i].astype('int')

    # 当一开始就 app_categories = app_categories[app_categories['appCategory'] != 0]
    # 经过查看，这时的数据中，在类别那块 appCategory_1 appCategory_2 都不存在为0类别的，至少也是从 1开始，所以可以直接对0进行替换
    # 将所有的0替换成 -1
    app_categories['appCategory_1'] = app_categories['appCategory_1'].replace(0,-1)
    app_categories['appCategory_2'] = app_categories['appCategory_2'].replace(0,-1)
    del app_categories['appCategory_new']
    app_categories.to_csv('../data/app_categories.csv' , index=False , mode='a' , encoding='utf-8')
    # >>> len(app_categories)
    # 217041
    # >>> len(app_categories['appID'].unique())
    # 217041


def a2():
    '''ad.csv'''
    # 这个文件就只查下：是否null就够了
    ad = pd.read_csv('../data/ago/ad.csv')
    for i in ad.columns:
        print  i,ad[i].isnull().sum()
    # creativeID 0
    # adID 0
    # camgaignID 0
    # advertiserID 0
    # appID 0
    # appPlatform 0

    # >>> len(ad)
    # 6582

    # >>> len(ad['appID'].unique())
    # 50
    # >>> len(ad['creativeID'].unique())
    # 6582

    # 原因是：一个广告下可以有多个素材，也就是1个appID下会有多个creativeID


def a3():
    '''user.csv'''
    # 1:查看下这个文件两列是否有空，最后两列指定下 为str类型
    user = pd.read_csv('../data/ago/user.csv',dtype={'hometown': 'int','residence': 'int'})
    user['hometown_new'] = user['hometown']
    user['residence_new'] = user['residence']
    for i in user.columns:
        print  i,user[i].isnull().sum()
    # userID 0
    # age 0
    # gender 0
    # education 0
    # marriageStatus 0
    # haveBaby 0
    # hometown 0
    # residence 0

    # hometown_new 若只有 3位，说明第一个数字是 一类特征、后两个数字是二类特征，若只有两个数字那就问题了，需要特别注意
    print (len(user[user['hometown_new'] < 100]))
    print (len(user[user['residence_new'] < 100]))

    user['hometown_new'] = user['hometown_new'].apply(lambda x : string.zfill(x,4))
    user['hometown_1'] = user['hometown_new'].str[:2]
    user['hometown_2'] = user['hometown_new'].str[2:]
    del user['hometown_new']

    user['residence_new'] = user['residence_new'].apply(lambda x : string.zfill(x,4))
    user['residence_1'] = user['residence_new'].str[:2]
    user['residence_2'] = user['residence_new'].str[2:]
    del user['residence_new']

    for i in user.columns:
        user[i] = user[i].astype('int')

    # 当一开始就 user = user[user['hometown'] != 0]   user = user[user['residence'] != 0]
    # >>> user['hometown_1'].unique()
    # array([ 5, 14,  6,  3, 16, 22, 24,  1, 10,  2,  4,  8,  7, 13,  9, 12, 17,
    #        28, 23, 11, 19, 26, 18, 15, 25, 21, 20, 27, 29, 30, 32, 33, 31, 34], dtype=int64)
    # >>> user['hometown_2'].unique()
    # array([12,  3,  7,  1, 13,  5, 16,  4, 10, 11,  8,  9,  6, 14,  2, 15, 19,
    #        17, 18, 21, 20], dtype=int64)
    # >>> user['residence_1'].unique()
    # array([ 5, 14,  6, 23,  3, 16, 22, 21, 18, 11, 10,  2,  8,  1, 13,  4,  9,
    #         7, 24, 17, 12, 28, 19, 15, 25, 26, 20, 27, 29, 30, 31, 33, 34, 32], dtype=int64)
    # >>> user['residence_2'].unique()
    # array([ 3,  7,  1, 13,  2,  6,  8, 12,  5,  9,  4, 14, 16, 20, 11, 10, 21,
    #        19, 17,  0, 15, 18], dtype=int64)
    # 只有 residence_2 存在 0这个类别，len(user[user['residence_2'] == 0]) == 3408 ，但是我想 这块的 0 应该也是代表未知吧，应该不是 具体的一个类别，因为
    # FAQ有解释：用户出生地，取值具体到市级城市，使用二级编码，千位百位数表示省份，十位个位数表示省内城市，如1806表示省份编号为18，城市编号是省内的6号，编号0表示未知。
    # 其余3个都没有0这个类别，单单这个有了？怪不！
    # 因此，之前的处理逻辑还是可以继续

    user['hometown_1'] = user['hometown_1'].replace(0,-1)
    user['hometown_2'] = user['hometown_2'].replace(0, -1)
    user['residence_1'] = user['residence_1'].replace(0,-1)
    user['residence_2'] = user['residence_2'].replace(0, -1)

    user.to_csv('../data/user.csv' , index=False , mode='a' , encoding='utf-8')

    # >> > len(user)
    # 2805118


def a4():
    # App安装列表(appInstallList)	截止到某一时间点用户全部的App安装列表(appID)，已过滤高频和低频App。
    # 这个文件解释的挺尴尬啊！！！  user_installedapps.csv

    user_installedapps = pd.read_csv('../data/ago/user_installedapps.csv')
    for i in user_installedapps.columns:
        print  i,user_installedapps[i].isnull().sum()
    # userID 0
    # appID 0

    print('len(user_installedapps[userID]) : ',len(user_installedapps['userID'].unique()))
    # ('len(user_installedapps[userID]) : ', 1446105)
# 比起上面的 2805118 来说，就是少多了！！！，但是这个文件不用做什么转换，

def a5():
    # 每行代表一个用户的单个App操作流水，各字段之间由逗号分隔，顺序依次为：“userID，installTime，appID”。特别的，
    # 我们提供了训练数据开始时间之前16天开始连续30天的操作流水，即第1天0点到第31天0点。
    user_app_actions = pd.read_csv('../data/ago/user_app_actions.csv')
    for i in user_app_actions.columns:
        print  i,user_app_actions[i].isnull().sum()
    # userID 0
    # installTime 0
    # appID 0

    print('len(user_app_actions[\'userID\'].unique()) : ',len(user_app_actions['userID'].unique()))
    # ("len(user_app_actions['userID'].unique()) : ", 781112)
    print('len(user_app_actions[\'appID\'].unique()) : ',len(user_app_actions['appID'].unique()))
    # ("len(user_app_actions['appID'].unique()) : ", 100923

    user_app_actions['installTime'] = user_app_actions['installTime'].astype(str).apply(lambda x: x.zfill(6))
    user_app_actions['clickTime_day'] = user_app_actions['installTime'].str[:2].astype('int')

    user_app_actions.to_csv('../data/ago/user_app_actions_new.csv' , index=False)



def a6():
    position = pd.read_csv('../data/ago/position.csv')
    for i in position.columns:
        print  i,position[i].isnull().sum()
    # positionID 0
    # sitesetID 0
    # positionType 0

def a7():
    train = pd.read_csv('../data/ago/train.csv')
    for i in train.columns:
        print  i,train[i].isnull().sum()
    # label 0
    # clickTime 0
    # conversionTime 3656266
    # creativeID 0
    # userID 0
    # positionID 0
    # connectionType 0
    # telecomsOperator 0
    # >>> len(train)
    # 3749528

    # >>> len(train[train['label'] == 1])
    # 93262
    # 意思就是训练集中：label这一列有 9万3262个

    # 将 conversionTime 列 fillna(-1)
    train['conversionTime'] = train['conversionTime'].fillna(-1)
    for i in train.columns:
        train[i] = train[i].astype('int')
    train.to_csv('../data/train.csv' , index=False , mode='a' , encoding='utf-8')



'''
函数说明：这个函数就是将多余的6个数据文件，merge到现有的训练集、测试集中
'''
def a8():
    print('mv to a8() function')
    '''---------------------------------------------探索下“训练集”-------------------------------------------------------'''
    train = pd.read_csv('../data/train.csv')
    train['instanceID'] = range(len(train))
    test = pd.read_csv('../data/test.csv')
    '''---------------------------打算要融合 app_categories ad--------------------------'''
    app_categories = pd.read_csv('../data/app_categories.csv')
    ad = pd.read_csv('../data/ad.csv')
    appcategories_ad_merge = pd.merge(ad, app_categories, on='appID', how='inner')
    appcategories_ad_merge.to_csv('../data/merge_data/appcategories_ad_merge.csv', index=False, mode='a',
                                  encoding='utf-8')
    train_merge_1 = pd.merge(train, appcategories_ad_merge, on='creativeID', how='left')
    train_merge_1.to_csv('../data/merge_data/train_merge_1.csv', index=False, mode='a', encoding='utf-8')
    test_merge_1 = pd.merge(test, appcategories_ad_merge, on='creativeID', how='left')
    test_merge_1.to_csv('../data/merge_data/test_merge_1.csv', index=False, mode='a', encoding='utf-8')

    '''----------------------------------打算要融合 user.csv、user_installedapps.csv、user_app_actions.csv----------------------------'''
    user = pd.read_csv('../data/user.csv')
    # user_installedapps = pd.read_csv('../data/user_installedapps.csv')
    # user_app_actions = pd.read_csv('../data/user_app_actions.csv')
    train_merge_2 = pd.merge(train_merge_1, user, on='userID', how='left')
    train_merge_2.to_csv('../data/merge_data/train_merge_2.csv', index=False, mode='a', encoding='utf-8')

    test_merge_2 = pd.merge(test_merge_1, user, on='userID', how='left')
    test_merge_2.to_csv('../data/merge_data/test_merge_2.csv', index=False, mode='a', encoding='utf-8')

    '''--------------------------------最后就是position 这个文件了------------------------------------'''
    position = pd.read_csv('../data/position.csv')
    train_merge_3 = pd.merge(train_merge_2, position, on='positionID', how='left')
    test_merge_3 = pd.merge(test_merge_2, position, on='positionID', how='left')
    train_merge_3.to_csv('../data/merge_data/train_merge_3.csv', index=False, mode='a', encoding='utf-8')
    test_merge_3.to_csv('../data/merge_data/test_merge_3.csv', index=False, mode='a', encoding='utf-8')


'''
函数说明：将 第一列时间那列--分成两个特征 小时、分钟
返回值：新的训练集、预测集
'''
def a9():
    print('mv to a9() function')
    train_merge_3 = pd.read_csv('../data/merge_data/train_merge_3.csv')
    test_merge_3 = pd.read_csv('../data/merge_data/test_merge_3.csv')

    # 删掉3个不需要的id列
    lie_will_delete = ['appCategory_2','hometown_2','residence_2']
    for i in lie_will_delete:
        del train_merge_3[i]
        del test_merge_3[i]

    # 在训练集、预测集中 的 installTime 列中，都是这样的数字：202359 210000，可以很容易的看出来，前两个数字是“天数”，中间两个是“小时”，最后两个是“秒数”
    test_merge_3['clickTime'] = test_merge_3['clickTime'].astype(str)
    test_merge_3['clickTime_day'] = test_merge_3['clickTime'].str[:2]
    test_merge_3['clickTime_day'] = test_merge_3['clickTime_day'].astype('int')
    test_merge_3['clickTime_hour'] = test_merge_3['clickTime'].str[2:4]
    test_merge_3['clickTime_hour'] = test_merge_3['clickTime_hour'].astype('int')

    train_merge_3['clickTime'] = train_merge_3['clickTime'].astype(str)

    train_merge_3['clickTime_day'] = train_merge_3['clickTime_day'].astype('int')
    train_merge_3['clickTime_hour'] = train_merge_3['clickTime'].str[2:4]
    train_merge_3['clickTime_hour'] = train_merge_3['clickTime_hour'].astype('int')



    # 做出来 clickTime_day 除以7 取余
    train_merge_3['clickTime_day_quyu'] = train_merge_3['clickTime_day']%7
    test_merge_3['clickTime_day_quyu'] = test_merge_3['clickTime_day'] % 7

    train_merge_3.to_csv('../data/merge_data/train_merge_4.csv', index=False, mode='a', encoding='utf-8')
    test_merge_3.to_csv('../data/merge_data/test_merge_4.csv', index=False, mode='a', encoding='utf-8')


'''
函数说明：这个函数是用来做每一个userID对应的 appCategory(unique后长度是14) appCategory_1(因为有缺失，所以unique后长度是5) 各安装了多少个？
        处理的是： user_installedapps.csv这个文件，做出来的数据集格式是 userID + 14列appCategory + 5列appCategory_1
函数不足：1：这个就是数据的不足了，并非在训练集中的每一个 userID 都有对应的 这几列特征，不足的我最终在把做出来的数据集添加到训练集、预测集的时候
            ，需要用 -99 来填充！
'''
def fea_1():
    user_installedapps = pd.read_csv('../data/user_installedapps.csv')
    # 查看下数据分布情况！
    train_merge_3 = pd.read_csv('../data/merge_data/train_merge_3.csv')
    print 'len(train_merge_3[\'userID\'].unique()) : ', len(train_merge_3['userID'].unique())
    # len(train_merge_3['userID'].unique()) :  2595627
    print 'len(user_installedapps[\'userID\'].unique()) : ', len(user_installedapps['userID'].unique())
    # len(user_installedapps['userID'].unique()) :  1446105
    print len(list(
        set(list(user_installedapps['userID'].unique())).intersection(set(list(train_merge_3['userID'].unique())))))
    # 1339593
    # 只有相交的一部分用户，那看来剩余的用户只能添加 -1 了

    # step2: 根据user_installedapps这个基础文件，做出来其中每一个用户的上述 2+6+9=17 个特征--首先是根据appID来添加 appPlatform appCategory_1 appCategory_2 这3列
    # >>> len(user_installedapps['appID'].unique())
    # 180389 可以看出来这个
    # 给 user_installedapps 根据 其中的 appID列，添加上 appPlatform appCategory_1 appCategory_2 这3列
    appcategories_ad_merge = pd.read_csv('../data/merge_data/appcategories_ad_merge.csv')
    # >>> len(appcategories_ad_merge['appID'].unique())
    # 50  这个里面只有50个appID，所以说配不上上面的 18万了！
    ad = pd.read_csv('../data/ad.csv')
    # >>> len(ad['appID'].unique())
    # 50 也配不上
    app_categories = pd.read_csv('../data/app_categories.csv')
    # >>> len(app_categories['appID'].unique())
    # 217041 这个有21万，倒是可以配上
    # 查看下 app_categories 和 user_installedapps 的appID的交集
    # >>> len(list(set(list(app_categories['appID'].unique())).intersection(set(list(user_installedapps['appID'].unique())))))
    # 180389

    # OK，appCategory_1 2可以添加上去了，但是appPlatform就不行了！
    # >>> len(app_categories)
    # 217041
    # 可以看出来 app_categories 中每一行就是一个app的，不存在重复的问题，所以可以直接将 app_categories user_installedapps进行merge

    # >>> len(user_installedapps)
    # 84039009
    user_installedapps = pd.merge(user_installedapps, app_categories, on='appID', how='left')

    # >>> len(user_installedapps)
    # 84039009
    # >>> user_installedapps.head()
    #    userID  appID  appCategory_1  appCategory_2
    # 0       1    357              2              1
    # 1       1    360              2              1
    # 2       1    362              4              9
    # 3       1    365              3              1
    # 4       1    375              2              3
    '''先做 appCategory_1'''
    user_installedapps_1 = user_installedapps.groupby(['userID', 'appCategory_1']).count()

    new = DataFrame()
    new['num'] = user_installedapps_1.reset_index(drop=True)['appID']
    # 2:第2,3列 用user_installedapps_1的index做出来
    linshi_1 = DataFrame(user_installedapps_1.index)
    linshi_1[0] = linshi_1[0].astype('str')
    # 这块的语法原因在哪？可以将其分成完全相同的3部分 .str.split('(').str[1].   str.split(')').str[0].   str.split(',').str[0]
    # linshi_1[0]是Series类型，没有str属性，所以先 str ，接着就可以split, 为啥要split两次呢，是因为上面 DataFrame(linshi.index)
    # 得到的形式是 (1,2)
    new['userID'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[0]
    new['appCategory_1'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[1]
    for i in new.columns:
        new[i] = new[i].astype('int')
    nnn = 0
    for i in new['userID'].unique():
        print(nnn)
        nnn += 1
        data = new[new['userID'] == i]
        # 先做出来一个 appCategory_1 列是“-1, 1, 2, 3, 4, 5”的DataFrame
        list_6 = [-1, 1, 2, 3, 4, 5]
        list_6 = DataFrame(list_6)
        list_6.columns = ['appCategory_1']

        new_data = pd.merge(list_6, data, on='appCategory_1', how='left')
        new_data['num'] = new_data['num'].fillna(0)
        new_data['num'] = new_data['num'].astype('int')
        new_data['userID'] = new_data['userID'].fillna(method='bfill')
        new_data['userID'] = new_data['userID'].fillna(method='ffill')
        # 做出来要写入的6个数字
        list_6_n = [int(new_data['userID'].unique()[0])]
        list_6_n += list(new_data['num'])
        list_6_n = DataFrame(list_6_n).T
        list_6_n.to_csv('../data/merge_data/appCategory_1.csv', index=False, encoding="utf-8", mode='a', header=False)

    '''再做 appCategory'''
    user_installedapps_1 = user_installedapps.groupby(['userID', 'appCategory']).count()

    new = DataFrame()
    new['num'] = user_installedapps_1.reset_index(drop=True)['appID']
    # 2:第2,3列 用user_installedapps_1的index做出来
    linshi_1 = DataFrame(user_installedapps_1.index)
    linshi_1[0] = linshi_1[0].astype('str')
    # 这块的语法原因在哪？可以将其分成完全相同的3部分 .str.split('(').str[1].   str.split(')').str[0].   str.split(',').str[0]
    # linshi_1[0]是Series类型，没有str属性，所以先 str ，接着就可以split, 为啥要split两次呢，是因为上面 DataFrame(linshi.index)
    # 得到的形式是 (1,2)
    new['userID'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[0]
    new['appCategory'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[1]
    for i in new.columns:
        new[i] = new[i].astype('int')
    nnn = 0
    for i in new['userID'].unique():
        print(nnn)
        nnn += 1
        data = new[new['userID'] == i]
        # 先做出来一个 time_stamp列是“-1, 1, 2, 3,4,5, 6, 7,8,9,10,11”的DataFrame
        list_6 = [108, 2, 209, 201, 402, 104, 301, 101, 203, 408, 106, 503, 0, 407]
        list_6 = DataFrame(list_6)
        list_6.columns = ['appCategory']

        new_data = pd.merge(list_6, data, on='appCategory', how='left')
        new_data['num'] = new_data['num'].fillna(0)
        new_data['num'] = new_data['num'].astype('int')
        new_data['userID'] = new_data['userID'].fillna(method='bfill')
        new_data['userID'] = new_data['userID'].fillna(method='ffill')
        # 做出来要写入的6个数字
        list_6_n = [int(new_data['userID'].unique()[0])]
        list_6_n += list(new_data['num'])
        list_6_n = DataFrame(list_6_n).T
        list_6_n.to_csv('../data/merge_data/appCategory_2.csv', index=False, encoding="utf-8", mode='a', header=False)

'''
函数说明：将 data/merge_data/下的 appCategory_1.csv _2.csv 两个文件添加到 train_merge_4 test_merge_4中去
注意点：肯定有将近一半的特征值要用 -99 来填充
'''
def a10():
    print('move to a10')
    train_merge_4 = pd.read_csv('../data/merge_data/train_merge_4.csv')
    test_merge_4 = pd.read_csv('../data/merge_data/test_merge_4.csv')
    clo_name_appCategory_1 = ['userID','count_cate_1_-1','count_cate_1_1','count_cate_1_2','count_cate_1_3','count_cate_1_4','count_cate_1_5']
    appCategory_1 = pd.read_csv('../data/merge_data/appCategory_1.csv', header=None, names=clo_name_appCategory_1)

    clo_name_appCategory_2 = ['userID']
    list_6 = [108, 2, 209, 201, 402, 104, 301, 101, 203, 408, 106, 503, 0, 407]
    for i in list_6:
        clo_name_appCategory_2.append('count_cate_'+str(i))
    appCategory_2 = pd.read_csv('../data/merge_data/appCategory_2.csv', header=None, names=clo_name_appCategory_2)

    train_merge_4 = pd.merge(train_merge_4, appCategory_1, on='userID', how='left')
    train_merge_4 = pd.merge(train_merge_4, appCategory_2, on='userID', how='left')
    test_merge_4 = pd.merge(test_merge_4, appCategory_1, on='userID', how='left')
    test_merge_4 = pd.merge(test_merge_4, appCategory_2, on='userID', how='left')
    for i in list(train_merge_4.columns)[-25:]:
        train_merge_4[i] = train_merge_4[i].fillna(-99)
        train_merge_4[i] = train_merge_4[i].astype('int')
        test_merge_4[i] = test_merge_4[i].fillna(-99)
        test_merge_4[i] = test_merge_4[i].astype('int')

    for i in train_merge_4.columns:
        print i,train_merge_4[i].isnull().sum()
    for i in train_merge_4.columns:
        print i,test_merge_4[i].isnull().sum()

    train_merge_4.to_csv('../data/merge_data/train_merge_5.csv', index=False, mode='a', encoding='utf-8')
    test_merge_4.to_csv('../data/merge_data/test_merge_5.csv', index=False, mode='a', encoding='utf-8')


'''
函数说明：17天到31天每个用户“每天”点击了多少次app
'''
def fea_2():
    print('mv to fea_2() function')
    train_merge_5 = pd.read_csv('../data/merge_data/train_merge_5.csv')
    test_merge_5 = pd.read_csv('../data/merge_data/test_merge_5.csv')

    '''先做训练集'''
    train_ad_group = train_merge_5.groupby(['clickTime_day','userID']).count()
    new = DataFrame()
    new['count_click'] = train_ad_group.reset_index(drop=True)['appID']
    # 第2,3列 用train_ad_group的index做出来
    linshi_1 = DataFrame(train_ad_group.index)
    linshi_1[0] = linshi_1[0].astype('str')
    new['clickTime_day'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[0]
    new['userID'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[1]

    # 接下来讲 new 和 train_merge_5 merge下，根据的是 userID clickTime_day
    train_merge_5['userID'] = train_merge_5['userID'].astype('int')
    train_merge_5['clickTime_day'] = train_merge_5['clickTime_day'].astype('int')
    new['userID'] = new['userID'].astype('int')
    new['clickTime_day'] = new['clickTime_day'].astype('int')
    train_merge_6 = pd.merge(train_merge_5, new, on=['userID','clickTime_day'], how='left')
    # 检查下是否有null，应该不会有
    print('fea_2 train null number : ',train_merge_6['count_click'].isnull().sum())
    train_merge_6.to_csv('../data/merge_data/train_merge_6.csv', index=False, mode='a', encoding='utf-8')

    '''再做测试集'''
    test_ad_group = test_merge_5.groupby(['clickTime_day', 'userID']).count()
    new = DataFrame()
    new['count_click'] = test_ad_group.reset_index(drop=True)['appID']
    # 2:第2,3列 用train_ad_group的index做出来
    linshi_1 = DataFrame(test_ad_group.index)
    linshi_1[0] = linshi_1[0].astype('str')
    new['clickTime_day'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[0]
    new['userID'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[1]
    # 接下来讲 new 和 train_merge_5 merge下，根据的是 userID clickTime_day
    test_merge_5['userID'] = test_merge_5['userID'].astype('int')
    test_merge_5['clickTime_day'] = test_merge_5['clickTime_day'].astype('int')
    new['userID'] = new['userID'].astype('int')
    new['clickTime_day'] = new['clickTime_day'].astype('int')
    test_merge_6 = pd.merge(test_merge_5, new, on=['userID', 'clickTime_day'], how='left')
    # 检查下是否有null，应该不会有
    print('fea_2 test null number : ', test_merge_6['count_click'].isnull().sum())
    test_merge_6.to_csv('../data/merge_data/test_merge_6.csv', index=False, mode='a', encoding='utf-8')

    # ('fea_2 train null number : ', 0)
    # ('fea_2 test null number : ', 0)



'''
这个特征使得成绩 从 0.122326 -》 0.122305
函数说明：提取每一个 appID 对应为1的label的样本数 / 对应为0的label的样本数
'''
def fea_bz():

    print('move to fea_bz() function')
    train_merge_5 = pd.read_csv('../data/merge_data/train_merge_5.csv')
    # >>> len(train) * 1.0 / len(train['userID'].unique())
    # 1.4445557855577862
    # 可以看出来 每一个用户只有 1.44 条数据，
    train_merge_5 = train_merge_5.groupby(['appID','label']).count()[['conversionTime']]
    new = DataFrame()
    new['label_count'] = train_merge_5.reset_index(drop=True)['conversionTime']
    # 2:第2,3列 用train_ad_group的index做出来
    linshi_1 = DataFrame(train_merge_5.index)
    linshi_1[0] = linshi_1[0].astype('str')
    new['appID'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[0].astype('int')
    new['label'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[1].astype('int')

    hang_50 = DataFrame()
    print('begin to make hang_50')
    for i in new['appID'].unique():
        print(i)
        data = new[new['appID'] == i]
        # 为了以防有的 appID 只有一种label
        list_2 = DataFrame([0,1])
        list_2.columns = ['label']
        merge_data = pd.merge(list_2, data, on='label', how='left')
        merge_data['label_count'] = merge_data['label_count'].fillna(0)
        merge_data['appID'] = merge_data['appID'].fillna(method='bfill')
        merge_data['appID'] = merge_data['appID'].fillna(method='ffill')
        one_hang = DataFrame(([i] + list(merge_data['label_count']))).T
        for j in one_hang.columns:
            one_hang[j] = one_hang[j].astype(int)
        hang_50 = hang_50.append(one_hang)
    hang_50.columns = ['appID','label_0','label_1']

    # 接下来就是求比值了，之前先 replace(0,1)
    hang_50['label_0'] = hang_50['label_0'].replace(0,1)
    hang_50['label1_div_label0'] = hang_50['label_1']*1.0 / hang_50['label_0']
    hang_50 = hang_50[['appID','label1_div_label0']]
    print('make hang_50 over')

    # 接下来就可以把 hang_50这个特征，添加到 train_merge_6 test_merge_6
    train_merge_6 = pd.read_csv('../data/merge_data/train_merge_6.csv')
    train_merge_6['appID'] = train_merge_6['appID'].astype('int')
    hang_50['appID'] = hang_50['appID'].astype('int')
    train_merge_7 = pd.merge(train_merge_6, hang_50, on=['appID'], how='left')
    print 'train null num : ',train_merge_7['label1_div_label0'].isnull().sum()
    train_merge_7.to_csv('../data/merge_data/train_merge_7.csv', index=False, mode='a', encoding='utf-8')

    test_merge_6 = pd.read_csv('../data/merge_data/test_merge_6.csv')
    test_merge_6['appID'] = test_merge_6['appID'].astype('int')
    test_merge_7 = pd.merge(test_merge_6, hang_50, on=['appID'], how='left')
    print 'test null num : ',test_merge_7['label1_div_label0'].isnull().sum()
    test_merge_7.to_csv('../data/merge_data/test_merge_7.csv', index=False, mode='a', encoding='utf-8')



'''
函数说明：做出来一个特征，是：这个用户在“当天”点击这个app“之前”是否安装过这个appID，0 1特征（先做出来所需要的字典，在a14中再做出来具体的特征）
函数思路：在下面可以看到被注释掉的一段程序，那段程序完全正确，就是耗费时间不可接受，所以把它注释掉，换成做一些字典出来
'''
def a13():

    # 先做出来3个文件对应的字典。user_installedapps对应一个字典就可以了；user_app_actions是30（1 12 123 。。。 123...30）
    # train_merge_7_label_equals_1是14 (17 1718 171819...30)
    train_merge_7 = pd.read_csv('../data/merge_data/train_merge_7.csv')
    train_merge_7['ago_label'] = 0
    # 加载能做出来这批字典的3个文件
    user_installedapps = pd.read_csv('../data/user_installedapps.csv')
    user_app_actions = pd.read_csv('../data/user_app_actions.csv')
    user_app_actions['installTime'] = user_app_actions['installTime'].astype(str).apply(lambda x : x.zfill(6))
    user_app_actions['installTime'] = user_app_actions['installTime'].astype(str).str[:2].astype('int')
    train_merge_7_label_equals_1 = train_merge_7[train_merge_7['label'] == 1]
    train_merge_7_label_equals_1 = train_merge_7_label_equals_1[['userID','appID','clickTime_day']]

    '''这块因为我有8个核，而且单核跑完整个1446105用户的话，太费时间，所以我将原本145万用户分成8部分，每部分180763.125，就是181000个用户'''
    '''最终花费时间在8小时左右'''
    # # 实际使用的方案
    # j = 1
    # n = 181000
    # dict_user_installedapps = {}
    # num = 0
    # for i in list(user_installedapps['userID'].unique())[(j-1)*n : j*n]:
    #     print num,1446105
    #     num += 1
    #     dict_user_installedapps[i] = list(user_installedapps[user_installedapps['userID'] == i]['appID'])
    #
    # cPickle.dump(dict_user_installedapps, open('../data/dicts/dict_user_installedapps_' + str(j) + '.pkl', 'wb'))

    # 如果有好机器使用的方案
    dict_user_installedapps = {}
    num = 0
    for i in list(user_installedapps['userID'].unique()):
        print num,1446105
        num += 1
        dict_user_installedapps[i] = list(user_installedapps[user_installedapps['userID'] == i]['appID'])

    cPickle.dump(dict_user_installedapps, open('../data/dicts/dict_user_installedapps.pkl', 'wb'))

    '''第二批字典做出来--每一天的字典，对应31个'''
    range_list = range(0, 31)
    for j in range_list:

        user_app_actions_copy = user_app_actions.copy()
        user_app_actions_copy = user_app_actions_copy[user_app_actions_copy['installTime'] <= j]
        dict_user_app_actions = {}
        num = 0
        for i in list(user_app_actions_copy['userID'].unique()):
            print j, num, 781112
            num += 1
            dict_user_app_actions[i] = list(user_app_actions_copy[user_app_actions_copy['userID'] == i]['appID'])

        cPickle.dump(dict_user_app_actions, open('../data/dicts/dict_user_app_actions_' + str(j) + '.pkl', 'wb'))

    # 实际使用的方案
    # '''之所以在第二批字典中，我还写了这样一段补充程序，是因为上面跑的太慢，把一些range_list中的天数移到这块来'''
    # range_list = [30]
    # for j in range_list:
    #
    #     user_app_actions_copy = user_app_actions[user_app_actions['installTime'] <= j]
    #     dict_user_app_actions = {}
    #     num = 0
    #     for i in list(user_app_actions_copy['userID'].unique())[int(len(user_app_actions_copy['userID'].unique()) * 0.5):]:
    #         print j, num, 781112
    #         num += 1
    #         dict_user_app_actions[i] = list(user_app_actions_copy[user_app_actions_copy['userID'] == i]['appID'])
    #
    #     cPickle.dump(dict_user_app_actions, open('../data/dicts/dict_user_app_actions_' + str(j) + '_2.pkl', 'wb'))


    '''第三批字典做出来--做出来17~30个字典'''
    range_list = range(17, 31)
    for j in range_list:

        train_merge_7_label_equals_1_copy = train_merge_7_label_equals_1[train_merge_7_label_equals_1['clickTime_day'] <= j]
        dict_train_merge_7_label_equals_1 = {}
        num = 0
        for i in list(train_merge_7_label_equals_1_copy['userID'].unique()):
            print j, num, 92051
            num += 1
            dict_train_merge_7_label_equals_1[i] = list(
                train_merge_7_label_equals_1_copy[train_merge_7_label_equals_1_copy['userID'] == i]['appID'])

        cPickle.dump(dict_train_merge_7_label_equals_1,
                     open('../data/dicts/dict_train_merge_7_label_equals_1_' + str(j) + '.pkl', 'wb'))



# def a13(): 跑起来很慢，而且根据想法 实际意义也不大，因为只给出来了31天的数据，一般在31天内大多数人类也不会重复安装2次app
#     train_merge_7 = pd.read_csv('../data/merge_data/train_merge_7.csv')
#     train_merge_7['ago_label'] = 0
#     # 做出来 看这个appID是否在 之前安装过的列表里面 的3个文件
#     user_installedapps = pd.read_csv('../data/user_installedapps.csv')
#     user_app_actions = pd.read_csv('../data/user_app_actions.csv')
#     user_app_actions['installTime'] = user_app_actions['installTime'].astype(str).str[:2].astype('int')
#     train_merge_7_label_equals_1 = train_merge_7[train_merge_7['label'] == 1]
#     train_merge_7_label_equals_1 = train_merge_7_label_equals_1[['userID','appID','clickTime_day']]
#
#     for i in range(0,len(train_merge_7)): # 根据索引找数据
#         print i
#         # 设置下这行数据的 ago_label 是否要进行改变？
#         data = train_merge_7[train_merge_7.index == i]
#         data_userID = data['userID'][i]
#         data_appID = data['appID'][i]
#         data_clickTime_day = data['clickTime_day'][i]
#         # 看这个appID是否在 之前安装过的列表里面；之前安装过的列表分成3个部分，一是user_installedapps.csv，二是user_app_actions.csv中 在这行记录data_clickTime_day之前的
#         # 三是：train_merge_7中在这行记录data_clickTime_day之前的 而且label还要为1
#         # >> > len(user_app_actions['userID'].unique())
#         # 781112
#         # >> > len(train_merge_7['userID'].unique())
#         # 2595627 可以看出来，user_app_actions要比train_merge_7中少很多用户
#
#         if( len(list(set(list(user_installedapps[user_installedapps['userID'] == data_userID]['appID'])).intersection(set([data_appID])))) > 0 ):
#             train_merge_7.loc[train_merge_7.index == i, 'ago_label'] = 1
#             continue
#
#         if (len(list(set(list(user_app_actions[user_app_actions['installTime'] < data_clickTime_day][user_app_actions[user_app_actions['installTime'] < data_clickTime_day]['userID'] == data_userID]['appID'])).intersection(set([data_appID])))) > 0):
#             train_merge_7.loc[train_merge_7.index == i, 'ago_label'] = 1
#             continue
#
#         if (len(list(set(list(train_merge_7_label_equals_1[train_merge_7_label_equals_1['clickTime_day'] < data_clickTime_day][train_merge_7_label_equals_1[train_merge_7_label_equals_1['clickTime_day'] < data_clickTime_day]['userID'] == data_userID]['appID'])).intersection(set([data_appID])))) > 0):
#             train_merge_7.loc[train_merge_7.index == i, 'ago_label'] = 1
#             continue
#
#     train_merge_7.to_csv('../data/merge_data/test_merge_9.csv', index=False, mode='a', encoding='utf-8')


'''
分数提高了万分位 ： 0.102305 -》 0.102253
函数说明：接着上面的a13做出来的所有字典，做出来这个特征的数值
    diaoyu_dict = cPickle.load(open('../data/dicts/diaoyu_dict.pkl', 'r'))
'''
def a14():
    # 实际使用的方案
    print('load first dicts')
    dict_train_merge_7_label_equals_1_17 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_17.pkl', 'r'))
    dict_train_merge_7_label_equals_1_18 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_18.pkl', 'r'))
    dict_train_merge_7_label_equals_1_19 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_19.pkl', 'r'))
    dict_train_merge_7_label_equals_1_20 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_20.pkl', 'r'))
    dict_train_merge_7_label_equals_1_21 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_21.pkl', 'r'))
    dict_train_merge_7_label_equals_1_22 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_22.pkl', 'r'))
    dict_train_merge_7_label_equals_1_23 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_23.pkl', 'r'))
    dict_train_merge_7_label_equals_1_24 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_24.pkl', 'r'))
    dict_train_merge_7_label_equals_1_25 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_25.pkl', 'r'))
    dict_train_merge_7_label_equals_1_26 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_26.pkl', 'r'))
    dict_train_merge_7_label_equals_1_27 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_27.pkl', 'r'))
    dict_train_merge_7_label_equals_1_28 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_28.pkl', 'r'))
    dict_train_merge_7_label_equals_1_29 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_29.pkl', 'r'))
    dict_train_merge_7_label_equals_1_30 = cPickle.load(open('../data/dicts/dict_train_merge_7_label_equals_1_30.pkl', 'r'))
    dict_train_merge_7_label_equals_1 = []
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_17)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_18)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_19)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_20)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_21)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_22)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_23)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_24)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_25)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_26)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_27)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_28)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_29)
    dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_30)

    print('load second dicts')
    dict_user_app_actions_1 = cPickle.load(open('../data/dicts/dict_user_app_actions_1.pkl', 'r'))
    dict_user_app_actions_2 = cPickle.load(open('../data/dicts/dict_user_app_actions_2.pkl', 'r'))
    dict_user_app_actions_3 = cPickle.load(open('../data/dicts/dict_user_app_actions_3.pkl', 'r'))
    dict_user_app_actions_4 = cPickle.load(open('../data/dicts/dict_user_app_actions_4.pkl', 'r'))
    dict_user_app_actions_5 = cPickle.load(open('../data/dicts/dict_user_app_actions_5.pkl', 'r'))
    dict_user_app_actions_6 = cPickle.load(open('../data/dicts/dict_user_app_actions_6.pkl', 'r'))
    dict_user_app_actions_7 = cPickle.load(open('../data/dicts/dict_user_app_actions_7.pkl', 'r'))
    dict_user_app_actions_8 = cPickle.load(open('../data/dicts/dict_user_app_actions_8.pkl', 'r'))
    dict_user_app_actions_9 = cPickle.load(open('../data/dicts/dict_user_app_actions_9.pkl', 'r'))
    dict_user_app_actions_10 = cPickle.load(open('../data/dicts/dict_user_app_actions_10.pkl', 'r'))
    dict_user_app_actions_11 = cPickle.load(open('../data/dicts/dict_user_app_actions_11.pkl', 'r'))
    dict_user_app_actions_12 = cPickle.load(open('../data/dicts/dict_user_app_actions_12.pkl', 'r'))
    dict_user_app_actions_13 = cPickle.load(open('../data/dicts/dict_user_app_actions_13.pkl', 'r'))
    dict_user_app_actions_14 = cPickle.load(open('../data/dicts/dict_user_app_actions_14.pkl', 'r'))
    dict_user_app_actions_15 = cPickle.load(open('../data/dicts/dict_user_app_actions_15.pkl', 'r'))
    dict_user_app_actions_16 = cPickle.load(open('../data/dicts/dict_user_app_actions_16.pkl', 'r'))
    dict_user_app_actions_17 = cPickle.load(open('../data/dicts/dict_user_app_actions_17.pkl', 'r'))
    dict_user_app_actions_18 = cPickle.load(open('../data/dicts/dict_user_app_actions_18.pkl', 'r'))
    dict_user_app_actions_19 = cPickle.load(open('../data/dicts/dict_user_app_actions_19.pkl', 'r'))
    dict_user_app_actions_20 = cPickle.load(open('../data/dicts/dict_user_app_actions_20.pkl', 'r'))
    dict_user_app_actions_21 = cPickle.load(open('../data/dicts/dict_user_app_actions_21.pkl', 'r'))
    dict_user_app_actions_22 = cPickle.load(open('../data/dicts/dict_user_app_actions_22.pkl', 'r'))
    dict_user_app_actions_23 = cPickle.load(open('../data/dicts/dict_user_app_actions_23.pkl', 'r'))
    dict_user_app_actions_24 = cPickle.load(open('../data/dicts/dict_user_app_actions_24.pkl', 'r'))
    dict_user_app_actions_25 = cPickle.load(open('../data/dicts/dict_user_app_actions_25.pkl', 'r'))
    dict_user_app_actions_26 = cPickle.load(open('../data/dicts/dict_user_app_actions_26.pkl', 'r'))
    dict_user_app_actions_27_1 = cPickle.load(open('../data/dicts/dict_user_app_actions_27_1.pkl', 'r'))
    dict_user_app_actions_27_2 = cPickle.load(open('../data/dicts/dict_user_app_actions_27_2.pkl', 'r'))
    dict_user_app_actions_27 = dict(dict_user_app_actions_27_1, **dict_user_app_actions_27_2)
    dict_user_app_actions_28_1 = cPickle.load(open('../data/dicts/dict_user_app_actions_28_1.pkl', 'r'))
    dict_user_app_actions_28_2 = cPickle.load(open('../data/dicts/dict_user_app_actions_28_2.pkl', 'r'))
    dict_user_app_actions_28 = dict(dict_user_app_actions_28_1, **dict_user_app_actions_28_2)
    dict_user_app_actions_29_1 = cPickle.load(open('../data/dicts/dict_user_app_actions_29_1.pkl', 'r'))
    dict_user_app_actions_29_2 = cPickle.load(open('../data/dicts/dict_user_app_actions_29_2.pkl', 'r'))
    dict_user_app_actions_29 = dict(dict_user_app_actions_29_1, **dict_user_app_actions_29_2)
    dict_user_app_actions_30_1 = cPickle.load(open('../data/dicts/dict_user_app_actions_30_1.pkl', 'r'))
    dict_user_app_actions_30_2 = cPickle.load(open('../data/dicts/dict_user_app_actions_30_2.pkl', 'r'))
    dict_user_app_actions_30 = dict(dict_user_app_actions_30_1, **dict_user_app_actions_30_2)
    dict_user_app_actions = []
    dict_user_app_actions.append(dict_user_app_actions_1)
    dict_user_app_actions.append(dict_user_app_actions_2)
    dict_user_app_actions.append(dict_user_app_actions_3)
    dict_user_app_actions.append(dict_user_app_actions_4)
    dict_user_app_actions.append(dict_user_app_actions_5)
    dict_user_app_actions.append(dict_user_app_actions_6)
    dict_user_app_actions.append(dict_user_app_actions_7)
    dict_user_app_actions.append(dict_user_app_actions_8)
    dict_user_app_actions.append(dict_user_app_actions_9)
    dict_user_app_actions.append(dict_user_app_actions_10)
    dict_user_app_actions.append(dict_user_app_actions_11)
    dict_user_app_actions.append(dict_user_app_actions_12)
    dict_user_app_actions.append(dict_user_app_actions_13)
    dict_user_app_actions.append(dict_user_app_actions_14)
    dict_user_app_actions.append(dict_user_app_actions_15)
    dict_user_app_actions.append(dict_user_app_actions_16)
    dict_user_app_actions.append(dict_user_app_actions_17)
    dict_user_app_actions.append(dict_user_app_actions_18)
    dict_user_app_actions.append(dict_user_app_actions_19)
    dict_user_app_actions.append(dict_user_app_actions_20)
    dict_user_app_actions.append(dict_user_app_actions_21)
    dict_user_app_actions.append(dict_user_app_actions_22)
    dict_user_app_actions.append(dict_user_app_actions_23)
    dict_user_app_actions.append(dict_user_app_actions_24)
    dict_user_app_actions.append(dict_user_app_actions_25)
    dict_user_app_actions.append(dict_user_app_actions_26)
    dict_user_app_actions.append(dict_user_app_actions_27)
    dict_user_app_actions.append(dict_user_app_actions_28)
    dict_user_app_actions.append(dict_user_app_actions_29)
    dict_user_app_actions.append(dict_user_app_actions_30)

    print('load third dicts')
    dict_user_installedapps_1 = cPickle.load(open('../data/dicts/dict_user_installedapps_1.pkl', 'r'))
    dict_user_installedapps_2 = cPickle.load(open('../data/dicts/dict_user_installedapps_2.pkl', 'r'))
    dict_user_installedapps_3 = cPickle.load(open('../data/dicts/dict_user_installedapps_3.pkl', 'r'))
    dict_user_installedapps_4 = cPickle.load(open('../data/dicts/dict_user_installedapps_4.pkl', 'r'))
    dict_user_installedapps_5 = cPickle.load(open('../data/dicts/dict_user_installedapps_5.pkl', 'r'))
    dict_user_installedapps_6 = cPickle.load(open('../data/dicts/dict_user_installedapps_6.pkl', 'r'))
    dict_user_installedapps_7 = cPickle.load(open('../data/dicts/dict_user_installedapps_7.pkl', 'r'))
    dict_user_installedapps_8 = cPickle.load(open('../data/dicts/dict_user_installedapps_8.pkl', 'r'))
    dict_user_installedapps = dict(dict_user_installedapps_1, **dict_user_installedapps_2)
    dict_user_installedapps = dict(dict_user_installedapps, **dict_user_installedapps_3)
    dict_user_installedapps = dict(dict_user_installedapps, **dict_user_installedapps_4)
    dict_user_installedapps = dict(dict_user_installedapps, **dict_user_installedapps_5)
    dict_user_installedapps = dict(dict_user_installedapps, **dict_user_installedapps_6)
    dict_user_installedapps = dict(dict_user_installedapps, **dict_user_installedapps_7)
    dict_user_installedapps = dict(dict_user_installedapps, **dict_user_installedapps_8)

    # # 将来提交代码时候，使用的方案 ，之后没注释的代码也得改下
    # print('load first dicts')
    # dict_train_merge_7_label_equals_1_17 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_17.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_18 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_18.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_19 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_19.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_20 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_20.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_21 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_21.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_22 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_22.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_23 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_23.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_24 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_24.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_25 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_25.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_26 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_26.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_27 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_27.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_28 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_28.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_29 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_29.pkl', 'r'))
    # dict_train_merge_7_label_equals_1_30 = cPickle.load(
    #     open('../data/dicts/dict_train_merge_7_label_equals_1_30.pkl', 'r'))
    # dict_train_merge_7_label_equals_1 = []
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_17)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_18)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_19)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_20)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_21)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_22)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_23)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_24)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_25)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_26)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_27)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_28)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_29)
    # dict_train_merge_7_label_equals_1.append(dict_train_merge_7_label_equals_1_30)
    #
    # print('load second dicts')
    # dict_user_app_actions_1 = cPickle.load(open('../data/dicts/dict_user_app_actions_1.pkl', 'r'))
    # dict_user_app_actions_2 = cPickle.load(open('../data/dicts/dict_user_app_actions_2.pkl', 'r'))
    # dict_user_app_actions_3 = cPickle.load(open('../data/dicts/dict_user_app_actions_3.pkl', 'r'))
    # dict_user_app_actions_4 = cPickle.load(open('../data/dicts/dict_user_app_actions_4.pkl', 'r'))
    # dict_user_app_actions_5 = cPickle.load(open('../data/dicts/dict_user_app_actions_5.pkl', 'r'))
    # dict_user_app_actions_6 = cPickle.load(open('../data/dicts/dict_user_app_actions_6.pkl', 'r'))
    # dict_user_app_actions_7 = cPickle.load(open('../data/dicts/dict_user_app_actions_7.pkl', 'r'))
    # dict_user_app_actions_8 = cPickle.load(open('../data/dicts/dict_user_app_actions_8.pkl', 'r'))
    # dict_user_app_actions_9 = cPickle.load(open('../data/dicts/dict_user_app_actions_9.pkl', 'r'))
    # dict_user_app_actions_10 = cPickle.load(open('../data/dicts/dict_user_app_actions_10.pkl', 'r'))
    # dict_user_app_actions_11 = cPickle.load(open('../data/dicts/dict_user_app_actions_11.pkl', 'r'))
    # dict_user_app_actions_12 = cPickle.load(open('../data/dicts/dict_user_app_actions_12.pkl', 'r'))
    # dict_user_app_actions_13 = cPickle.load(open('../data/dicts/dict_user_app_actions_13.pkl', 'r'))
    # dict_user_app_actions_14 = cPickle.load(open('../data/dicts/dict_user_app_actions_14.pkl', 'r'))
    # dict_user_app_actions_15 = cPickle.load(open('../data/dicts/dict_user_app_actions_15.pkl', 'r'))
    # dict_user_app_actions_16 = cPickle.load(open('../data/dicts/dict_user_app_actions_16.pkl', 'r'))
    # dict_user_app_actions_17 = cPickle.load(open('../data/dicts/dict_user_app_actions_17.pkl', 'r'))
    # dict_user_app_actions_18 = cPickle.load(open('../data/dicts/dict_user_app_actions_18.pkl', 'r'))
    # dict_user_app_actions_19 = cPickle.load(open('../data/dicts/dict_user_app_actions_19.pkl', 'r'))
    # dict_user_app_actions_20 = cPickle.load(open('../data/dicts/dict_user_app_actions_20.pkl', 'r'))
    # dict_user_app_actions_21 = cPickle.load(open('../data/dicts/dict_user_app_actions_21.pkl', 'r'))
    # dict_user_app_actions_22 = cPickle.load(open('../data/dicts/dict_user_app_actions_22.pkl', 'r'))
    # dict_user_app_actions_23 = cPickle.load(open('../data/dicts/dict_user_app_actions_23.pkl', 'r'))
    # dict_user_app_actions_24 = cPickle.load(open('../data/dicts/dict_user_app_actions_24.pkl', 'r'))
    # dict_user_app_actions_25 = cPickle.load(open('../data/dicts/dict_user_app_actions_25.pkl', 'r'))
    # dict_user_app_actions_26 = cPickle.load(open('../data/dicts/dict_user_app_actions_26.pkl', 'r'))
    # dict_user_app_actions_27 = cPickle.load(open('../data/dicts/dict_user_app_actions_27.pkl', 'r'))
    # dict_user_app_actions_28 = cPickle.load(open('../data/dicts/dict_user_app_actions_28.pkl', 'r'))
    # dict_user_app_actions_29 = cPickle.load(open('../data/dicts/dict_user_app_actions_29.pkl', 'r'))
    # dict_user_app_actions_30 = cPickle.load(open('../data/dicts/dict_user_app_actions_30.pkl', 'r'))
    # dict_user_app_actions = []
    # dict_user_app_actions.append(dict_user_app_actions_1)
    # dict_user_app_actions.append(dict_user_app_actions_2)
    # dict_user_app_actions.append(dict_user_app_actions_3)
    # dict_user_app_actions.append(dict_user_app_actions_4)
    # dict_user_app_actions.append(dict_user_app_actions_5)
    # dict_user_app_actions.append(dict_user_app_actions_6)
    # dict_user_app_actions.append(dict_user_app_actions_7)
    # dict_user_app_actions.append(dict_user_app_actions_8)
    # dict_user_app_actions.append(dict_user_app_actions_9)
    # dict_user_app_actions.append(dict_user_app_actions_10)
    # dict_user_app_actions.append(dict_user_app_actions_11)
    # dict_user_app_actions.append(dict_user_app_actions_12)
    # dict_user_app_actions.append(dict_user_app_actions_13)
    # dict_user_app_actions.append(dict_user_app_actions_14)
    # dict_user_app_actions.append(dict_user_app_actions_15)
    # dict_user_app_actions.append(dict_user_app_actions_16)
    # dict_user_app_actions.append(dict_user_app_actions_17)
    # dict_user_app_actions.append(dict_user_app_actions_18)
    # dict_user_app_actions.append(dict_user_app_actions_19)
    # dict_user_app_actions.append(dict_user_app_actions_20)
    # dict_user_app_actions.append(dict_user_app_actions_21)
    # dict_user_app_actions.append(dict_user_app_actions_22)
    # dict_user_app_actions.append(dict_user_app_actions_23)
    # dict_user_app_actions.append(dict_user_app_actions_24)
    # dict_user_app_actions.append(dict_user_app_actions_25)
    # dict_user_app_actions.append(dict_user_app_actions_26)
    # dict_user_app_actions.append(dict_user_app_actions_27)
    # dict_user_app_actions.append(dict_user_app_actions_28)
    # dict_user_app_actions.append(dict_user_app_actions_29)
    # dict_user_app_actions.append(dict_user_app_actions_30)
    #
    # print('load third dicts')
    # dict_user_installedapps = cPickle.load(open('../data/dicts/dict_user_installedapps.pkl', 'r'))



    train_merge_7 = pd.read_csv('../data/merge_data/train_merge_7.csv')
    train_merge_7['ago_label'] = 0
    for i in range(0,len(train_merge_7)): # 根据索引找数据
        print i
        # 设置下这行数据的 ago_label 是否要进行改变？
        data = train_merge_7[train_merge_7.index == i]
        data_userID = data['userID'][i]
        data_appID = data['appID'][i]
        data_clickTime_day = data['clickTime_day'][i]
        # 看这个appID是否在 之前安装过的列表里面；之前安装过的列表分成3个部分，一是user_installedapps.csv，二是user_app_actions.csv中 在这行记录data_clickTime_day之前的
        # 三是：train_merge_7中在这行记录data_clickTime_day之前的 而且label还要为1
        # >> > len(user_app_actions['userID'].unique())
        # 781112
        # >> > len(train_merge_7['userID'].unique())
        # 2595627 可以看出来，user_app_actions要比train_merge_7中少很多用户

        try:
            if( len(list(set(dict_user_installedapps[data_userID]).intersection(set([data_appID])))) > 0 ):
                train_merge_7.loc[train_merge_7.index == i, 'ago_label'] = 1
                continue
        except:
            ll = 1

        # 因为我上面做 dict_user_app_actions 这个的时候，是用 <= 天数的逻辑来的，所以每次要 减去1，再加上索引是从0开始的，所以就再减去1
        try:
            if( len(list(set(dict_user_app_actions[data_clickTime_day - 2][data_userID]).intersection(set([data_appID])))) > 0):
                train_merge_7.loc[train_merge_7.index == i, 'ago_label'] = 1
                continue
        except:
            ll = 1

        try:
            if(data_clickTime_day >= 18):
                if( len(list(set(dict_train_merge_7_label_equals_1[data_clickTime_day - 18][data_userID]).intersection(set([data_appID])))) > 0):
                    train_merge_7.loc[train_merge_7.index == i, 'ago_label'] = 1
                    continue
        except:
            ll = 1

    train_merge_7.to_csv('../data/merge_data/train_merge_8.csv', index=False, mode='a', encoding='utf-8')

    train_merge_7 = pd.read_csv('../data/merge_data/test_merge_7.csv')
    train_merge_7['ago_label'] = 0
    for i in range(0, len(train_merge_7)):  # 根据索引找数据
        print i
        # 设置下这行数据的 ago_label 是否要进行改变？
        data = train_merge_7[train_merge_7.index == i]
        data_userID = data['userID'][i]
        data_appID = data['appID'][i]
        data_clickTime_day = data['clickTime_day'][i]
        # 看这个appID是否在 之前安装过的列表里面；之前安装过的列表分成3个部分，一是user_installedapps.csv，二是user_app_actions.csv中 在这行记录data_clickTime_day之前的
        # 三是：train_merge_7中在这行记录data_clickTime_day之前的 而且label还要为1
        # >> > len(user_app_actions['userID'].unique())
        # 781112
        # >> > len(train_merge_7['userID'].unique())
        # 2595627 可以看出来，user_app_actions要比train_merge_7中少很多用户

        try:
            if (len(list(set(dict_user_installedapps[data_userID]).intersection(set([data_appID])))) > 0):
                train_merge_7.loc[train_merge_7.index == i, 'ago_label'] = 1
                continue
        except:
            ll = 1

        # 因为我上面做 dict_user_app_actions 这个的时候，是用 <= 天数的逻辑来的，所以每次要 减去1，再加上索引是从0开始的，所以就再减去1
        try:
            if (len(list(set(dict_user_app_actions[data_clickTime_day - 2][data_userID]).intersection(
                    set([data_appID])))) > 0):
                train_merge_7.loc[train_merge_7.index == i, 'ago_label'] = 1
                continue
        except:
            ll = 1

        try:
            if (data_clickTime_day >= 18):
                if (len(list(set(dict_train_merge_7_label_equals_1[data_clickTime_day - 18][data_userID]).intersection(
                        set([data_appID])))) > 0):
                    train_merge_7.loc[train_merge_7.index == i, 'ago_label'] = 1
                    continue
        except:
            ll = 1

    train_merge_7.to_csv('../data/merge_data/test_merge_8.csv', index=False, mode='a', encoding='utf-8')




'''
这个函数添加的历史转化率成绩确实提升很大。B榜上1200次迭代，从0.104839-》0.102998
函数说明：所有类别中 除了userID外，其他的都提些类别特征的历史转换率，不过这次不能有数据穿越，所以详细的函数步骤我得记录下！
        因为一批csv文件是 29个，为了不数据穿越，我这次先试下统计前1周的id转换情况，那天数 17 18 19 20 21 22 23前面都不足1周，所以最终的训练集
        不涉及到这些天数的数据！我只用做 24 25 26 27 28 29 30 31 天一周前的历史转化率
补充说明：这个函数和zhuanhua_chuan()这个函数意义是一模一样的，有一点不同就是：zhuanhua_chuan()中统计的是 1~30天的转化率，
        本函数每次统计的 前1周 中的转化率，生成的文件数和zhuanhua_chuan()也一模一样，但是多了一列就是 时间，时间列unique一下就是 24~31
'''
def zhuanhua_not_chuan():
    print('mv to zhuanhua_not_chuan function')
    # 计算下面这些列中，每一个列的 历史转化率
    col = ['creativeID', 'userID', 'positionID', 'connectionType', 'telecomsOperator', 'adID', 'camgaignID',
           'advertiserID', 'appID', 'appPlatform', 'appCategory','clickTime_day','clickTime_day_quyu',
           'appCategory_1', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown',
           'hometown_1',  'label',
           'residence', 'residence_1', 'sitesetID', 'positionType', 'clickTime_hour']
    train_merge_8 = pd.read_csv('../data/merge_data/train_merge_8.csv')
    train_merge_8['clickTime_day_quyu'] = train_merge_8['clickTime_day'] % 7

    train_merge_8 = train_merge_8[col]
    col.remove('userID')
    col.remove('label')
    col.remove('clickTime_day')

    for i in col:
        print i
        for k in [24,25,26,27,28,29,30,31]:
            train_merge_8_copy = train_merge_8[train_merge_8['clickTime_day'] < k]
            train_merge_8_copy = train_merge_8_copy[train_merge_8_copy['clickTime_day'] >= k-7]
            train_merge_groupby = train_merge_8_copy.groupby([i, 'label']).count()[['userID']]

            new = DataFrame()
            new['zhuan_count'] = train_merge_groupby.reset_index(drop=True)['userID']
            # 2:第2,3列 用train_ad_group的index做出来
            linshi_1 = DataFrame(train_merge_groupby.index)
            linshi_1[0] = linshi_1[0].astype('str')
            new[i] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[0]
            new['label'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[1]
            for ii in new.columns:
                new[ii] = new[ii].astype('int')

            # new_1 存储了每一个 creativeID 的历史转化率
            new_1 = DataFrame()
            new_1[i] = new[i].unique()
            new_1[i+'_bulv'] = 0  # 先初始为0，后面用程序自行改变
            num = len(new_1[i].unique())
            for j in new_1[i].unique():
                print(i, j, num)
                data = new[new[i] == j]
                try:
                    label_1 = list(data.ix[data['label'] == 1, 'zhuan_count'])[0]
                except:
                    label_1 = 0
                try:
                    label_0 = list(data.ix[data['label'] == 0, 'zhuan_count'])[0]
                except:
                    label_0 = 0
                new_1.loc[new_1[i] == j, i+'_bulv'] = label_1 * 1.0 / (label_1 + label_0)
            # 经过上面那个for循环，每一个creativeID的历史转化率就做好了！输出到磁盘上，将来每个数据文件都可以用
            new_1['clickTime_day'] = k
            if(k == 24):
                new_1.to_csv('../data/zhuanhualv_all/' + i + '.csv', index=False, encoding="utf-8" , mode='a')
            else:
                new_1.to_csv('../data/zhuanhualv_all/' + i + '.csv', index=False, encoding="utf-8", mode='a',header=None)


'''
函数说明：这个函数是要把刚才形成的一些 历史转化率数据 连接在训练集、测试集上
'''
def a16():
    print('mv to a16 function')
    path = '../data/zhuanhualv_all/'
    files = os.listdir(path)

    file_all = []
    for i in files:
        j = pd.read_csv(path + i)
        j.columns = [list(j.columns)[0] , list(j.columns)[0] + '_' + list(j.columns)[1] ,list(j.columns)[2] ]
        file_all.append(j)

    train_merge_8 = pd.read_csv('../data/merge_data/train_merge_8.csv')
    train_merge_8['clickTime_day_quyu'] = train_merge_8['clickTime_day'] % 7
    for i in [17,18,19,20,21,22,23]:
        train_merge_8 = train_merge_8[train_merge_8['clickTime_day'] != i]

    test_merge_8 = pd.read_csv('../data/merge_data/test_merge_8.csv')
    test_merge_8['clickTime_day_quyu'] = test_merge_8['clickTime_day'] % 7

    # for循环每次添加一个文件
    num = 0
    for i in file_all:
        print i.head()
        # 每一个转化率文件就是从 train_merge_7 中统计7天出来的，并不一定会涉及到 每一个属性，所以还得有个fillna(-99)
        train_merge_8 = pd.merge(train_merge_8 , i , on=[files[num][:-4] , 'clickTime_day'] , how='left')
        test_merge_8 = pd.merge(test_merge_8 , i , on=[files[num][:-4] , 'clickTime_day'] , how='left')
        # 预测集中也一样
        train_merge_8[list(i.columns)[1]] = train_merge_8[list(i.columns)[1]].fillna(-99)
        test_merge_8[list(i.columns)[1]] = test_merge_8[list(i.columns)[1]].fillna(-99)
        num += 1
    for i in train_merge_8.columns:
        print 'train',i,train_merge_8[i].isnull().sum()
    for i in test_merge_8.columns:
        print 'test',i,test_merge_8[i].isnull().sum()
    train_merge_8.to_csv('../data/merge_data/train_merge_9.csv',index=False)
    test_merge_8.to_csv('../data/merge_data/test_merge_9.csv',index=False)


'''
函数说明：把 user_action_app 这个数据文件，7天时间窗，求出来用户在这段时间内 appCategory对应属性 appCategory一级类目对应属性 这个userID分别各安装了多少
'''
def a18():
    print('mv to a18 function')
    user_app_actions = pd.read_csv('../data/user_app_actions_new.csv')
    del user_app_actions['installTime']
    app_categories = pd.read_csv('../data/app_categories.csv')
    del app_categories['appCategory_2']
    user_app_actions = pd.merge(user_app_actions, app_categories, on='appID', how='left')

    '''先做 appCategory '''
    for k in [24, 25, 26, 27, 28, 29, 30, 31]:
        user_app_actions_copy = user_app_actions[user_app_actions['clickTime_day'] < k]
        user_app_actions_copy = user_app_actions_copy[user_app_actions_copy['clickTime_day'] >= k - 7]
        user_installedapps_appCategory = user_app_actions_copy.groupby(['userID', 'appCategory']).count()

        new = DataFrame()
        new['appCategory_num'] = user_installedapps_appCategory.reset_index(drop=True)['appID']
        # 2:第2,3列 用user_installedapps_1的index做出来
        linshi_1 = DataFrame(user_installedapps_appCategory.index)
        linshi_1[0] = linshi_1[0].astype('str')

        new['userID'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[0]
        new['appCategory'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[1]
        for i in new.columns:
            new[i] = new[i].astype('int')
        # >>> new.head()
        #    appCategory_num userID appCategory
        # 0               37      1           0
        # 1                1      1         103
        # 2                2      1         108
        # 3                9      1         201
        # 4                5      1         203

        all_num = len(new['userID'].unique())
        nnn = 0
        for i in new['userID'].unique():
            print k,nnn, all_num
            nnn += 1
            data = new[new['userID'] == i]
            # 先做出来一个 appCategory_1 列是“下面那些”的DataFrame
            list_6 = [201, 409, 301, 203, 503, 407, 0, 103, 406, 209, 108, 211, 402, 210, 2, 405, 408, 106, 403, 401, 109,
                      104, 303, 110, 105, 204, 1]
            list_6 = DataFrame(list_6)
            list_6.columns = ['appCategory']

            new_data = pd.merge(list_6, data, on='appCategory', how='left')
            new_data['appCategory_num'] = new_data['appCategory_num'].fillna(0).astype('int')
            new_data['userID'] = new_data['userID'].fillna(method='bfill')
            new_data['userID'] = new_data['userID'].fillna(method='ffill')
            # 做出来要写入的27个数字
            list_6_n = [i,k]
            list_6_n += list(new_data['appCategory_num'])
            list_6_n = DataFrame(list_6_n).T

            list_6_n.to_csv('../data/merge_data/shixiong_user_actions_appCategory.csv', index=False, encoding="utf-8", mode='a',
                            header=False)

        '''再做 appCategory_1 '''
        user_installedapps_appCategory = user_app_actions_copy.groupby(['userID', 'appCategory_1']).count()

        new = DataFrame()
        new['appCategory_1_num'] = user_installedapps_appCategory.reset_index(drop=True)['appID']
        # 2:第2,3列 用user_installedapps_1的index做出来
        linshi_1 = DataFrame(user_installedapps_appCategory.index)
        linshi_1[0] = linshi_1[0].astype('str')

        new['userID'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[0]
        new['appCategory_1'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[1]

        for i in new.columns:
            new[i] = new[i].astype('int')

        nnn = 0
        for i in new['userID'].unique():
            print k,nnn, all_num
            nnn += 1
            data = new[new['userID'] == i]
            # 先做出来一个 appCategory_1 列是“下面那些”的DataFrame
            list_6 = [2, 4, 3, 5, -1, 1]
            list_6 = DataFrame(list_6)
            list_6.columns = ['appCategory_1']

            new_data = pd.merge(list_6, data, on='appCategory_1', how='left')
            new_data['appCategory_1_num'] = new_data['appCategory_1_num'].fillna(0).astype('int')
            new_data['userID'] = new_data['userID'].fillna(method='bfill')
            new_data['userID'] = new_data['userID'].fillna(method='ffill')
            # 做出来要写入的27个数字
            list_6_n = [i,k]
            list_6_n += list(new_data['appCategory_1_num'])
            list_6_n = DataFrame(list_6_n).T
            list_6_n.to_csv('../data/merge_data/shixiong_user_actions_appCategory_1.csv', index=False, encoding="utf-8", mode='a',
                            header=False)


'''
函数说明：一个trick 重复样本(userID+clickTime相同)的两个特征。1：在当天重复的次数，是一列特征，多为1；  2：rank排序的时候，就是从1开始 2 3 4这样子
'''
def DupFun(trainData):
    dicts = {}
    for i in range(trainData.index[0], trainData.index[0] + len(trainData)):
        print i
        key = trainData.ix[i, 'userID'].astype(str) + trainData.ix[i, 'clickTime'].astype(str)
        if (dicts.get(key) == None):
            dicts[key] = []
        dicts[key].append(trainData.ix[i, 'instanceID'])

    # 重复数量统计
    dupList = []
    rankList = []
    for i in range(trainData.index[0], trainData.index[0] + len(trainData)):
        print i
        key = trainData.ix[i, 'userID'].astype(str) + trainData.ix[i, 'clickTime'].astype(str)
        lens = len(dicts.get(key))
        if (lens == 1):  # 此时没有重复，那么排序为0
            rankList.append(0)
        else:
            num = 0
            for j in dicts.get(key):
                num = num + 1
                if (trainData.ix[i, 'instanceID'] == j):
                    rankList.append(num)

        # print lens
        dupList.append(lens)
    print len(rankList)
    ss = Series(dupList)
    ss.index = range(trainData.index[0], trainData.index[0] + len(trainData))
    trainData['DupNum'] = ss

    tt = Series(rankList)
    tt.index = range(trainData.index[0], trainData.index[0] + len(trainData))
    trainData['RankNum'] = tt
    return trainData
def a19():
    train_merge_10 = pd.read_csv('../data/merge_data/train_merge_9.csv')
    test_merge_10 = pd.read_csv('../data/merge_data/test_merge_9.csv')

    train_merge_10 = DupFun(train_merge_10)
    test_merge_10 = DupFun(test_merge_10)

    train_merge_10.to_csv('../data/merge_data/train_merge_11.csv' , index=False )
    test_merge_10.to_csv('../data/merge_data/test_merge_11.csv', index=False)




'''
函数说明：将上面 user_actions 做出来的两个数据文件，添加进 train_merge_11 中，输出 train_merge_12
'''
def a18_1():
    print('mv to a18_1 function')
    train_merge_9 = pd.read_csv('../data/merge_data/train_merge_11.csv')
    test_merge_9 = pd.read_csv('../data/merge_data/test_merge_11.csv')
    print('len(train_merge_9) : ', len(train_merge_9))
    print('len(test_merge_9) : ', len(test_merge_9))
    shixiong_appCategory = pd.read_csv('../data/merge_data/shixiong_user_actions_appCategory.csv', header=None)
    list_6 = [201, 409, 301, 203, 503, 407, 0, 103, 406, 209, 108, 211, 402, 210, 2, 405, 408, 106, 403, 401, 109, 104,
              303, 110, 105, 204, 1]
    for i in range(len(list_6)):
        list_6[i] = 'user_actions_appCategory_' + str(list_6[i])
    shixiong_appCategory_col = ['userID','clickTime_day'] + list_6
    shixiong_appCategory.columns = shixiong_appCategory_col

    shixiong_appCategory_1 = pd.read_csv('../data/merge_data/shixiong_user_actions_appCategory_1.csv', header=None)
    list_6 = [2, 4, 3, 5, -1, 1]
    for i in range(len(list_6)):
        list_6[i] = 'user_actions_appCategory_1_' + str(list_6[i])
    shixiong_appCategory_1_col = ['userID','clickTime_day'] + list_6
    shixiong_appCategory_1.columns = shixiong_appCategory_1_col

    train_merge_9 = pd.merge(train_merge_9, shixiong_appCategory, on=['userID','clickTime_day'], how='left')
    train_merge_9 = pd.merge(train_merge_9, shixiong_appCategory_1, on=['userID','clickTime_day'], how='left')
    for i in train_merge_9.columns:
        train_merge_9[i] = train_merge_9[i].fillna(-99)

    test_merge_9 = pd.merge(test_merge_9, shixiong_appCategory, on=['userID','clickTime_day'], how='left')
    test_merge_9 = pd.merge(test_merge_9, shixiong_appCategory_1, on=['userID','clickTime_day'], how='left')
    for i in test_merge_9.columns:
        test_merge_9[i] = test_merge_9[i].fillna(-99)
    print('write')
    print('len(train_merge_9) : ', len(train_merge_9))
    print('len(test_merge_9) : ', len(test_merge_9))
    train_merge_9.to_csv('../data/merge_data/train_merge_12.csv', index=False)
    test_merge_9.to_csv('../data/merge_data/test_merge_12.csv', index=False)



'''
这个函数是放在：师兄服务器上进行的
函数说明：做出来11个id特征的两两交集，的转化率！
'''
def a22():
    print('mv to a22 function')
    train_merge_10 = pd.read_csv('../data/merge_data/train_merge_8.csv')
    col = ['appID','age','creativeID','positionID','adID','camgaignID','userID','residence','advertiserID',
           'connectionType','clickTime']
    col_all = []
    for i in col:
        print i
        for j in col:
            if(i == j):
                a = 1
            else:
                ii = i + '_' + j
                train_merge_10[ii] = train_merge_10[i].astype('str') + train_merge_10[j].astype('str')
                train_merge_10[ii] = train_merge_10[ii].astype('int')
                col_all.append(ii)

    # 然后提取 转化率
    print('begin to zuhe_zhuanhualv')
    # 计算下面这些列中，每一个列的 历史转化率
    for i in col_all:
        for k in [24, 25, 26, 27, 28, 29, 30, 31]:
            print i,k
            train_merge_8_copy = train_merge_10[train_merge_10['clickTime_day'] < k]
            train_merge_8_copy = train_merge_8_copy[train_merge_8_copy['clickTime_day'] >= k - 7]
            train_merge_groupby = train_merge_8_copy.groupby([i, 'label']).count()[['userID']]

            new = DataFrame()
            new['zhuan_count'] = train_merge_groupby.reset_index(drop=True)['userID']
            # 2:第2,3列 用train_ad_group的index做出来
            linshi_1 = DataFrame(train_merge_groupby.index)
            linshi_1[0] = linshi_1[0].astype('str')
            new[i] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[0]
            new['label'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[1]
            for ii in new.columns:
                new[ii] = new[ii].astype('int')

            new_0 = new[new['label'] == 0]
            new_1 = new[new['label'] == 1]
            new_0.columns = ['label_0_count',i,'label_0']
            new_1.columns = ['label_1_count',i,'label_1']
            new_0_1 = pd.merge(new_0 , new_1 , on=i ,how='outer')
            del new_0_1['label_0']
            del new_0_1['label_1']
            new_0_1['label_0_count'] = new_0_1['label_0_count'].fillna(0)
            new_0_1['label_1_count'] = new_0_1['label_1_count'].fillna(0)
            # 做出来 转化率
            new_0_1[i+'_bilv'] = new_0_1['label_1_count'] / (new_0_1['label_0_count'] + new_0_1['label_1_count'])

            new_0_1 = new_0_1[[i , i+'_bilv']]
            new_0_1['clickTime_day'] = k

            if (k == 24):
                new_0_1.to_csv('../data/zhuanhualv_zuhe/' + i + '.csv', index=False, encoding="utf-8", mode='a')
            else:
                new_0_1.to_csv('../data/zhuanhualv_zuhe/' + i + '.csv', index=False, encoding="utf-8", mode='a', header=None)

            # # new_1 存储了每一个 creativeID 的历史转化率
            # new_1 = DataFrame()
            # new_1[i] = new[i].unique()
            # new_1[i + 'bulv'] = 0  # 先初始为0，后面用程序自行改变
            # num = len(new_1[i].unique())
            # for j in new_1[i].unique():
            #     print(i, j, num)
            #     data = new[new[i] == j]
            #     try:
            #         label_1 = list(data.ix[data['label'] == 1, 'zhuan_count'])[0]
            #     except:
            #         label_1 = 0
            #     try:
            #         label_0 = list(data.ix[data['label'] == 0, 'zhuan_count'])[0]
            #     except:
            #         label_0 = 0
            #     new_1.loc[new_1[i] == j, i + 'bulv'] = label_1 * 1.0 / (label_1 + label_0)
            # # 经过上面那个for循环，每一个creativeID的历史转化率就做好了！输出到磁盘上，将来每个数据文件都可以用
            # new_1['clickTime_day'] = k
            # if (k == 24):
            #     new_1.to_csv('../data/zhuanhualv_zuhe/' + i + '.csv', index=False, encoding="utf-8", mode='a')
            # else:
            #     new_1.to_csv('../data/zhuanhualv_zuhe/' + i + '.csv', index=False, encoding="utf-8", mode='a',
            #                  header=None)



'''
函数说明：这个函数是按照师兄的说明，进一步扩展之前说的那个trick。
'DupNum_userID' ： 在整个集合中，这个userID有多少条样本
'trick_userID_RankNum' ： 对重复的样本做一个排序
'trick_userID_cha_time' ： 计算下每一个相同样本和第一条样本的时间差
'''
# 计算t1 t2之间的差！t1 t2的时间格式都是 170003这样的格式 t2>t1
def a24_0(t1 , t2):
    a = int(str(t2)[:2]) - int(str(t1)[:2])
    b = int(str(t2)[2:4]) - int(str(t1)[2:4])
    # c = int(str(t2)[-2:]) - int(str(t1)[-2:])
    return a*24 + b

def a24_1(trainData):
    dicts = {}
    instance_time = {} # 存储每一个instanceID对应样本的时间（clickTime）！
    for i in range(trainData.index[0], trainData.index[0] + len(trainData)):
        print 1,i
        key = trainData.ix[i, 'userID'].astype(str)
        if (dicts.get(key) == None):
            dicts[key] = []
        dicts[key].append(trainData.ix[i, 'instanceID'])
        instance_time[trainData.ix[i, 'instanceID']] = trainData.ix[i, 'clickTime']

    # 做出来一个字典--索引是每一个userID，值是这个userID的首次样本出现时间
    dicts_2 = {}
    for i, j in dicts.iteritems():
        print 2,i
        dicts_2[int(i)] = instance_time[j[0]]

    rankList = []
    rankList_time = []
    dupList = [] # 用于计算同一userID有多少条数据
    for i in range(trainData.index[0], trainData.index[0] + len(trainData)):
        print 3,i
        key = trainData.ix[i, 'userID'].astype(str)
        lens = len(dicts.get(key))
        if (lens == 1):  # 此时没有重复，那么排序为0
            rankList.append(0)
            rankList_time.append(0)
        else:
            num = 0
            for j in dicts.get(key):
                num = num + 1
                if (trainData.ix[i, 'instanceID'] == j):
                    rankList.append(num)
                    # 给 rankList_time 添加上时间差！
                    rankList_time.append(        a24_0(dicts_2[trainData.ix[i, 'userID']] , trainData.ix[i, 'clickTime'])        )
        dupList.append(lens)
    print 'len(rankList) : ',len(rankList)
    print 'len(rankList_time) : ',len(rankList_time)
    print 'len(dupList) : ',len(dupList)

    tt = Series(rankList)
    ttt = Series(rankList_time)
    ss = Series(dupList)
    tt.index = range(trainData.index[0], trainData.index[0] + len(trainData))
    ttt.index = range(trainData.index[0], trainData.index[0] + len(trainData))
    ss.index = range(trainData.index[0], trainData.index[0] + len(trainData))

    trainData['DupNum_userID'] = ss
    trainData['trick_userID_RankNum'] = tt
    trainData['trick_userID_cha_time'] = ttt
    return trainData

def a24_2():
    train_merge_5 = pd.read_csv('../data/merge_data/train_merge_5.csv')
    train_merge_5 = train_merge_5[['instanceID' , 'clickTime_day' , 'userID','clickTime']]
    test_merge_5 = pd.read_csv('../data/merge_data/test_merge_5.csv')
    test_merge_5 = test_merge_5[['instanceID', 'clickTime_day', 'userID','clickTime']]

    train_merge_5 = train_merge_5.append(test_merge_5)
    train_merge_5 = train_merge_5.reset_index(drop=True)
    train_merge_5['instanceID'] = train_merge_5.index.astype(int) + 1


    train_merge_5 = a24_1(train_merge_5)


    for i in [17,18,19,20,21,22,23]:
        train_merge_5 = train_merge_5[train_merge_5['clickTime_day'] != i]

    train_merge_12 = pd.read_csv('../data/merge_data/train_merge_12.csv')
    train_merge_12 = train_merge_12[train_merge_12['clickTime_day'] != 23]
    train_merge_12['trick_userID_RankNum'] = list(train_merge_5.head(len(train_merge_12))['trick_userID_RankNum'])
    train_merge_12['trick_userID_cha_time'] = list(train_merge_5.head(len(train_merge_12))['trick_userID_cha_time'])
    train_merge_12['DupNum_userID'] = list(train_merge_5.head(len(train_merge_12))['DupNum_userID'])

    test_merge_12 = pd.read_csv('../data/merge_data/test_merge_12.csv')
    test_merge_12['trick_userID_RankNum'] = list(train_merge_5.tail(len(test_merge_12))['trick_userID_RankNum'])
    test_merge_12['trick_userID_cha_time'] = list(train_merge_5.tail(len(test_merge_12))['trick_userID_cha_time'])
    test_merge_12['DupNum_userID'] = list(train_merge_5.tail(len(test_merge_12))['DupNum_userID'])

    print(len(train_merge_12))
    print(len(test_merge_12))

    train_merge_12.to_csv('../data/merge_data/train_merge_13.csv', index=False)
    test_merge_12.to_csv('../data/merge_data/test_merge_13.csv', index=False)



'''
函数说明：试下选手公布出来的那张表上的21个组合特征
'''
def a25():
    print('mv to a25 function')
    train_merge_10 = pd.read_csv('../data/merge_data/train_merge_8.csv')
    col_all = ['positionID_connectionType', 'advertiserID_positionID', 'gender_positionID' , 'hometown_residence', 'age_marriageStatus',
              'age_positionID', 'appID_positionID', 'hometown_positionID', 'telecomsOperator_positionID', 'creativeID_positionID',
              'education_positionID', 'camgaignID_positionID', 'age_creativeID', 'marriageStatus_positionID', 'residence_positionID',
              'age_telecomsOperator', 'age_education', 'camgaignID_connectionType' , 'gender_education', 'marriageStatus_residence'
              ]
    for ii in col_all:
        print ii
        i = ii.split('_')[0]
        j = ii.split('_')[1]
        train_merge_10[ii] = train_merge_10[i].astype('str') + train_merge_10[j].astype('str')
        train_merge_10[ii] = train_merge_10[ii].astype('int')

    # 然后提取 转化率
    print('begin to a25 zhuanhualv')
    # 计算下面这些列中，每一个列的 历史转化率
    for i in col_all:
        for k in [24, 25, 26, 27, 28, 29, 30, 31]:
            print i,k
            train_merge_8_copy = train_merge_10[train_merge_10['clickTime_day'] < k]
            train_merge_8_copy = train_merge_8_copy[train_merge_8_copy['clickTime_day'] >= k - 7]
            train_merge_groupby = train_merge_8_copy.groupby([i, 'label']).count()[['userID']]

            new = DataFrame()
            new['zhuan_count'] = train_merge_groupby.reset_index(drop=True)['userID']
            # 2:第2,3列 用train_ad_group的index做出来
            linshi_1 = DataFrame(train_merge_groupby.index)
            linshi_1[0] = linshi_1[0].astype('str')
            new[i] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[0]
            new['label'] = linshi_1[0].str.split('(').str[1].str.split(')').str[0].str.split(',').str[1]
            for ii in new.columns:
                new[ii] = new[ii].astype('int')

            new_0 = new[new['label'] == 0]
            new_1 = new[new['label'] == 1]
            new_0.columns = ['label_0_count',i,'label_0']
            new_1.columns = ['label_1_count',i,'label_1']
            new_0_1 = pd.merge(new_0 , new_1 , on=i ,how='outer')
            del new_0_1['label_0']
            del new_0_1['label_1']
            new_0_1['label_0_count'] = new_0_1['label_0_count'].fillna(0)
            new_0_1['label_1_count'] = new_0_1['label_1_count'].fillna(0)
            # 做出来 转化率
            new_0_1[i+'_bilv'] = new_0_1['label_1_count'] / (new_0_1['label_0_count'] + new_0_1['label_1_count'])

            new_0_1 = new_0_1[[i , i+'_bilv']]
            new_0_1['clickTime_day'] = k

            if (k == 24):
                new_0_1.to_csv('../data/zhuanhualv_zuhe/' + i + '.csv', index=False, encoding="utf-8", mode='a')
            else:
                new_0_1.to_csv('../data/zhuanhualv_zuhe/' + i + '.csv', index=False, encoding="utf-8", mode='a', header=None)
# 配套上面 把 这些转化率文件 merge 到 训练集、测试集 上
def a26():
    print('mv to a26 function')
    path = '../data/zhuanhualv_zuhe/'
    files = os.listdir(path)
    files.remove('zhuanhualv_zuhe')
    files.remove('123.zip')

    file_all = []
    for i in files:
        j = pd.read_csv(path + i)
        file_all.append(j)

    train_merge_8 = pd.read_csv('../data/merge_data/train_merge_13.csv')
    test_merge_8 = pd.read_csv('../data/merge_data/test_merge_13.csv')

    col_all = ['positionID_connectionType', 'advertiserID_positionID', 'gender_positionID', 'hometown_residence',
               'age_marriageStatus',
               'age_positionID', 'appID_positionID', 'hometown_positionID', 'telecomsOperator_positionID',
               'creativeID_positionID',
               'education_positionID', 'camgaignID_positionID', 'age_creativeID', 'marriageStatus_positionID',
               'residence_positionID',
               'age_telecomsOperator', 'age_education', 'camgaignID_connectionType', 'gender_education',
               'marriageStatus_residence'
               ]
    for ii in col_all:
        print ii
        i = ii.split('_')[0]
        j = ii.split('_')[1]
        train_merge_8[ii] = train_merge_8[i].astype('str') + train_merge_8[j].astype('str')
        train_merge_8[ii] = train_merge_8[ii].astype('int')
        test_merge_8[ii] = test_merge_8[i].astype('str') + test_merge_8[j].astype('str')
        test_merge_8[ii] = test_merge_8[ii].astype('int')

    # for循环每次添加一个文件
    num = 0
    for i in file_all:
        print i.head()
        # 每一个转化率文件就是从 train_merge_7 中统计7天出来的，并不一定会涉及到 每一个属性，所以还得有个fillna(-99)
        train_merge_8 = pd.merge(train_merge_8 , i , on=[files[num][:-4] , 'clickTime_day'] , how='left')
        test_merge_8 = pd.merge(test_merge_8 , i , on=[files[num][:-4] , 'clickTime_day'] , how='left')
        # 预测集中也一样
        train_merge_8[list(i.columns)[1]] = train_merge_8[list(i.columns)[1]].fillna(-99)
        test_merge_8[list(i.columns)[1]] = test_merge_8[list(i.columns)[1]].fillna(-99)
        num += 1
    for i in train_merge_8.columns:
        print 'train',i,train_merge_8[i].isnull().sum()
    for i in test_merge_8.columns:
        print 'test',i,test_merge_8[i].isnull().sum()
    train_merge_8.to_csv('../data/merge_data/train_merge_14.csv',index=False)
    test_merge_8.to_csv('../data/merge_data/test_merge_14.csv',index=False)



if __name__ == "__main__":
    a1()
    a2()
    a3()
    a4()
    a5()
    a6()
    a7()

    a8()
    a9()
    fea_1()
    a10()
    fea_2()  # train_merge_6
    fea_bz()  # train_merge_7
    a13()
    a14()  # train_merge_8
    zhuanhua_not_chuan()
    a16()  # train_merge_9
    installed_fea()
    a17()  # train_merge_10

    a18() # 做出来 user_action 这个文件的两个 数据文件

    a19() # 添加上师兄说的 trick 的两个特征！ train_merge_11

    a22()

    a18_1() # 添加上 user_action的几个特征-》train_merge_12

    a23()

    a24_2() # trick_userID_RankNum  trick_userID_cha_time  DupNum_userID

    a25()
    a26()