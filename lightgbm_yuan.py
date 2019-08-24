# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:57:23 2019

@author: Administrator
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from datetime import timedelta
import pandas as pd
import numpy as np
import time
from sklearn.metrics import f1_score
from sklearn.feature_selection import chi2, SelectPercentile
import xgboost as xgb
import lightgbm as lgb
import gc
import warnings
warnings.filterwarnings('ignore')


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df




#读入数据
traindata = pd.read_csv("data/round1_iflyad_anticheat_traindata.txt",sep='\t')
testdata = pd.read_csv("data/round1_iflyad_anticheat_testdata_feature.txt",sep='\t')
label = traindata.pop('label')
test_id = testdata['sid'].values


#将训练集与测试集进行组合
data=pd.concat([traindata,testdata],axis=0).reset_index(drop=True)


########特征工程###########
#sid最后有一个时间，把他提取出来，并且和请求服务到达时间做减法
data['begin_time']=data['sid'].apply(lambda x:int(x.split('-')[-1])) ##请求会话时间
#data['nginxtime-begin_time']=data['nginxtime']-data['begin_time']##请求会话时间 与 请求到达服务时间的差


#针对nginxtime做特征工程
data['date'] = pd.to_datetime(data['nginxtime'] / 1000, unit='s') + timedelta(hours=8)
data['hour'] = data['date'].dt.hour.astype('int')
data['minute'] = data['date'].dt.minute.astype('int')
data['day'] = data['date'].dt.day.astype('int')#训练集是3到9天 测试集是第10天
data['dayofweek'] = data['date'].dt.dayofweek.astype('int')


#将设备高和设备宽组合成设备的面积
data['area'] = data['h'] * data['w']
data['creative_dpi'] = data['w'].astype(str) + "_" + data['h'].astype(str)


#idfamd5缺失严重，可以删除
data.drop(['idfamd5'], axis=1, inplace=True)

#把openudidmd5列按缺失与否转换成1和0
data.loc[data['openudidmd5']=='empty', 'openudidmd5'] = 0
data.loc[data['openudidmd5']!=0, 'openudidmd5'] = 1

#缺失值处理
for col in ['city','lan','make','model','osv','ver']:
    data[col].fillna('null_value', inplace=True)
    


# orientation 出现异常值 90度和2 归为 0
data.orientation[(data.orientation == 90) | (data.orientation == 2)] = 0

# carrier  -1 就是0
data.carrier[data.carrier == -1] = 0

#os只有两个，Android和android,这两类与标签比例不同。转换为0和1，也可以亚编码
data.loc[data['os']=='android', 'os'] = 0
data.loc[data['os']=='Android', 'os'] = 1


# 运营商 carrier
data.ntt[(data.carrier <= 0) | (data.carrier > 46003)] = 0


#osv
def osv_fix(x):
    if 'Android_' in x:
        x = x.strip('Android_')
    if 'Android' in x:
        x = x.strip('Android')
    return x

data['osv'] = data['osv'].astype('str')
data['osv'] = data['osv'].apply(osv_fix)

# make
def make_fix(x):
    x = x.lower()
    if 'iphone' in x or 'apple' in x:
        return 'apple'
    if '华为' in x or 'huawei' in x or "荣耀" in x:
        return 'huawei'
    if "魅族" in x:
        return 'meizu'
    if "金立" in x:
        return 'gionee'
    if "三星" in x:
        return 'samsung'
    if 'xiaomi' in x or 'redmi' or '小米' in x:
        return 'xiaomi'
    if 'oppo' in x:
        return 'oppo'
    return x

data['make'] = data['make'].astype('str').apply(lambda x: x.lower())
data['make'] = data['make'].apply(make_fix)


#删除无用的特征
data.drop(['sid','nginxtime','begin_time'],axis=1, inplace=True)
gc.collect()

#编码，加速
categorical_features = ['pkgname','ver','adunitshowid','mediashowid','ip','city','province',
    'reqrealip','adidmd5','imeimd5','macmd5','model','make','osv','lan','creative_dpi']

for col in categorical_features:
    print(col)
    data[col] = data[col].map(
        dict(zip(data[col].unique(), range(0, data[col].nunique()))))

gc.collect()

##################聚合特征和统计特征####################

    
media_cate_feature = ['pkgname','ver','adunitshowid','mediashowid','apptype']   
ip_cate_feature = ['ip','city','province','reqrealip'] 
model_cate_feature = ['adidmd5', 'imeimd5',  'macmd5','dvctype','model','make','ntt','carrier','osv','creative_dpi']
    
#将一些特征的出现次数作为特征,命名为vc_+特证名
value_counts_col = media_cate_feature+ip_cate_feature+model_cate_feature
count_feature_list = []
def feature_count(data, features=[], is_feature=True):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    nunique = []
    for i in features:
        nunique.append(data[i].nunique())
        new_feature += '_' + i.replace('add_', '')
    if len(features) > 1 and len(data[features].drop_duplicates()) <= np.max(nunique):
        print(new_feature, 'is unvalid cross feature:')
        return data
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
    data = data.merge(temp, 'left', on=features)
    if is_feature:
        count_feature_list.append(new_feature)
    return data


for i in value_counts_col:
    print(i)
    n = data[i].nunique()
    if n > 5:
        data = feature_count(data, [i])
        data = feature_count(data, ['day', 'hour', i])    
    
gc.collect()    

   
ratio_feature_list = []
for i in ['adunitshowid']:
    for j in ['city','model','reqrealip','ip','ver','osv','creative_dpi']:
        print(j)
        data = feature_count(data, [i, j])
        if data[i].nunique() > 5 and data[j].nunique() > 5:
            data['ratio_' + j + '_of_' + i] = data['count_' + i + '_' + j] / data['count_' + i]
            data['ratio_' + i + '_of_' + j] = data['count_' + i + '_' + j] / data['count_' + j]
            ratio_feature_list.append('ratio_' + j + '_of_' + i)
            ratio_feature_list.append('ratio_' + i + '_of_' + j)
    
# 低频过滤
for feature in value_counts_col:
    if 'count_' + feature in data.keys():
        print(feature)
        data.loc[data['count_' + feature] < 2, feature] = -1
        data[feature] = data[feature] + 1


    
def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))
        
    
predictors = []    
'''
ip = ip
app = adunitshowid
device = adidmd5
os = model
channel = mediashowid
click_time = nginxtime
date 
'''
#计算之后的时间差    
def cal_next_time_delta(df, suffix, type='float32'):
    groupby_columns = [
        {'columns': ['model']},
       {'columns': ['city']},
        {'columns': ['adunitshowid']},
       {'columns': ['reqrealip']},
       {'columns': ['creative_dpi']},
       {'columns': ['adunitshowid']},
       {'columns': ['ip']},
       {'columns': ['model','adunitshowid']},
       {'columns': ['model','creative_dpi']},
       {'columns': ['mediashowid']},
       {'columns': ['model', 'city']},
    ]
    # Calculate the time to next click for each group
    for spec in groupby_columns:
        # Name of new feature
        new_name = '{}_{}'.format('_'.join(spec['columns']), suffix)
        # Unique list of features to select
        all_features = spec['columns'] + ['date']
        # Run calculation
        log('Calculate ' + suffix + '...')
        df[new_name] = (df[all_features].groupby(spec['columns']).date.shift(-1) - df.date).dt.seconds.astype(type)
        predictors.append(new_name)
        gc.collect()
    return df    
    
log('Cal next_time_delta')
data = cal_next_time_delta(data, 'next_time_delta', 'float32')    
    
#计算之前的时间差    
def cal_prev_time_delta(df, suffix, type='float32'):
    groupby_columns = [
        {'columns': ['model']},
       {'columns': ['city']},
        {'columns': ['adunitshowid']},
       {'columns': ['reqrealip']},
       {'columns': ['creative_dpi']},
       {'columns': ['adunitshowid']},
       {'columns': ['ip']},
       {'columns': ['model','adunitshowid']},
       {'columns': ['model','creative_dpi']},
       {'columns': ['mediashowid']},
       {'columns': ['model', 'city']},
    ]
    # Calculate the time to prev click for each group
    for spec in groupby_columns:
        # Name of new feature
        new_name = '{}_{}'.format('_'.join(spec['columns']), suffix)
        # Unique list of features to select
        all_features = spec['columns'] + ['date']
        # Run calculation
        log('Calculate ' + suffix + '...')
        df[new_name] = (df.date - df[all_features].groupby(spec['columns']).date.shift(+1)).dt.seconds.astype(type)
        predictors.append(new_name)
        gc.collect()
    return df    
    
log('Cal prev_time_delta')
data = cal_prev_time_delta(data, 'prev_time_delta', 'float32')    
    
#删除无用的特征
data.drop(['date'],axis=1, inplace=True)

#计算唯一值特征    
def merge_nunique(df, columns_groupby, column, new_column_name, type='uint64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].nunique()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df    

log('Cal nunique_model_gb_ip')
data = merge_nunique(data, ['ip'], 'model', 'nunique_model_gb_ip', 'uint32')
gc.collect()    
    
log('Cal nunique_adunitshowid_gb_ip')
data = merge_nunique(data, ['ip'], 'adunitshowid', 'nunique_adunitshowid_gb_ip', 'uint32')
gc.collect() 

log('Cal nunique_city_gb_model')
data = merge_nunique(data, ['model'], 'city', 'nunique_city_gb_model', 'uint32')
gc.collect()     

log('Cal nunique_reqrealip_gb_model')
data = merge_nunique(data, ['model'], 'reqrealip', 'nunique_reqrealip_gb_model', 'uint32')
gc.collect()  

log('Cal nunique_adunitshowid_gb_model')
data = merge_nunique(data, ['model'], 'adunitshowid', 'nunique_adunitshowid_gb_model', 'uint32')
gc.collect()  

log('Cal nunique_model_gb_adunitshowid')
data = merge_nunique(data, ['adunitshowid'], 'model', 'nunique_model_gb_adunitshowid', 'uint32')
gc.collect()   

log('Cal nunique_model_gb_reqrealip')
data = merge_nunique(data, ['reqrealip'], 'model', 'nunique_model_gb_reqrealip', 'uint32')
gc.collect()    
    
log('Cal nunique_reqrealip_gb_adunitshowid')
data = merge_nunique(data, ['adunitshowid'], 'reqrealip', 'nunique_reqrealip_gb_adunitshowid', 'uint32')
gc.collect()   

log('Cal nunique_ip_gb_reqrealip')
data = merge_nunique(data, ['reqrealip'], 'ip', 'nunique_ip_gb_reqrealip', 'uint32')
gc.collect()  

log('Cal nunique_hour_gb_model_day')
data = merge_nunique(data, ['model', 'day'], 'hour', 'nunique_hour_gb_model_day', 'uint32')
gc.collect()

log('Cal nunique_hour_gb_city_day')
data = merge_nunique(data, ['city', 'day'], 'hour', 'nunique_hour_gb_city_day', 'uint32')
gc.collect()

log('Cal nunique_day_gb_ip_adunitshowid')
data = merge_nunique(data, ['reqrealip', 'day'], 'hour', 'nunique_day_gb_reqrealip_day', 'uint32')
gc.collect()
    

#计算cumcount特征
def merge_cumcount(df, columns_groupby, column, new_column_name, type='uint64'):
    df[new_column_name] = df.groupby(columns_groupby)[column].cumcount().values.astype(type)
    predictors.append(new_column_name)
    return df

log('Cal cumcount_city_gb_model')
data = merge_cumcount(data, ['model'], 'city', 'cumcount_city_gb_model', 'uint32');
gc.collect()

log('Cal cumcount_reqrealip_gb_model')
data = merge_cumcount(data, ['model'], 'reqrealip', 'cumcount_reqrealip_gb_model', 'uint32');
gc.collect()

log('Cal cumcount_adunitshowid_gb_model')
data = merge_cumcount(data, ['model'], 'adunitshowid', 'cumcount_adunitshowid_gb_model', 'uint32');
gc.collect()

#计算count特征
def merge_count(df, columns_groupby, new_column_name, type='uint64'):
    add = pd.DataFrame(df.groupby(columns_groupby).size()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df

log('Cal count_gb_model_city')
data = merge_count(data, ['model', 'city'], 'count_gb_model_city', 'uint32');
gc.collect()

log('Cal count_gb_model_reqrealip')
data = merge_count(data, ['model', 'reqrealip'], 'count_gb_model_reqrealip', 'uint32');
gc.collect()

log('Cal count_gb_model_adunitshowid')
data = merge_count(data, ['model', 'adunitshowid'], 'count_gb_model_adunitshowid', 'uint32');
gc.collect()

log('Cal count_gb_model_ip')
data = merge_count(data, ['model', 'ip'], 'count_gb_model_ip', 'uint32');
gc.collect()

log('Cal count_gb_city_reqrealip')
data = merge_count(data, ['city', 'reqrealip'], 'count_gb_city_reqrealip', 'uint32');
gc.collect()

log('Cal count_gb_city_adunitshowid')
data = merge_count(data, ['city', 'adunitshowid'], 'count_gb_city_adunitshowid', 'uint32');
gc.collect()

log('Cal count_gb_reqrealip_adunitshowid')
data = merge_count(data, ['reqrealip', 'adunitshowid'], 'count_gb_reqrealip_adunitshowid', 'uint32');
gc.collect()

log('Cal count_gb_reqrealip_ip')
data = merge_count(data, ['reqrealip', 'ip'], 'count_gb_reqrealip_ip', 'uint32');
gc.collect()

log('Cal count_gb_city_hour')
data = merge_count(data, ['city', 'hour'], 'count_gb_city_hour', 'uint32');
gc.collect()

log('Cal count_gb_model_hour')
data = merge_count(data, ['model', 'hour'], 'count_gb_model_hour', 'uint32');
gc.collect()


#计算方差特征
def merge_var(df, columns_groupby, column, new_column_name, type='float64'):
    add = pd.DataFrame(df.groupby(columns_groupby)[column].var()).reset_index()
    add.columns = columns_groupby + [new_column_name]
    df = df.merge(add, on=columns_groupby, how="left")
    df[new_column_name] = df[new_column_name].astype(type)
    predictors.append(new_column_name)
    return df

log('Cal var_day_gb_model')
data = merge_var(data, ['model'], 'day', 'var_day_gb_model', 'float32')
gc.collect()

log('Cal var_hour_gb_model')
data = merge_var(data, ['model'], 'hour', 'var_hour_gb_model', 'float32')
gc.collect()

log('Cal var_day_gb_city')
data = merge_var(data, ['city'], 'day', 'var_day_gb_city', 'float32')
gc.collect()

log('Cal var_hour_gb_city')
data = merge_var(data, ['city'], 'hour', 'var_hour_gb_city', 'float32')
gc.collect()

log('Cal var_day_gb_reqrealip')
data = merge_var(data, ['reqrealip'], 'day', 'var_day_gb_reqrealip', 'float32')
gc.collect()

log('Cal var_hour_gb_reqrealip')
data = merge_var(data, ['reqrealip'], 'hour', 'var_hour_gb_reqrealip', 'float32')
gc.collect()

log('Cal var_day_gb_adunitshowid')
data = merge_var(data, ['adunitshowid'], 'day', 'var_day_gb_adunitshowid', 'float32')
gc.collect()

log('Cal var_hour_gb_mediashowid')
data = merge_var(data, ['mediashowid'], 'hour', 'var_hour_gb_mediashowid', 'float32')
gc.collect()


data.fillna(-1, inplace=True)




#求一些特征列中每个特征出现的时间的最大值，最小值以及差值
# =============================================================================
# day_col = ['ver', 'adunitshowid', 'ip', 'reqrealip', 'model']
# 
# def data_group(data, col):
#     #day
#     df = pd.DataFrame()
#     df['early_day_' + col] = data.groupby(col)['day'].min()
#     df['later_day_' + col] = data.groupby(col)['day'].max()
#     df['range_day_' + col] = df['later_day_' + col] - df['early_day_' + col]
#     
#     df = df.reset_index()
#     return df    
# 
# for col in day_col:
#     print(col)
#     group_data = data_group(data, col)
#     data = pd.merge(data, group_data, on=col, how='left')
# 
# =============================================================================

gc.collect()


#减少内存的使用
data = reduce_mem_usage(data)


cat_feature = value_counts_col+['openudidmd5','os','orientation','lan']
num_feature = [col for col in data.columns if col not in cat_feature]


#建立countVector特征
data['new_con'] = data['model'].astype(str)
for i in ['city', 'reqrealip','adunitshowid','mediashowid']:
    data['new_con'] = data['new_con'].astype(str) + '_' + data[i].astype(str)
data['new_con'] = data['new_con'].apply(lambda x: ' '.join(x.split('_')))



##划分数据：
train=data[:traindata.shape[0]]
test=data[traindata.shape[0]:]
predict_result = pd.DataFrame()
predict_result['sid'] = test_id
predict_result['label'] = 0
train_y = label.values

base_train_csr = sparse.csr_matrix((len(train), 0))
base_predict_csr = sparse.csr_matrix((len(test), 0))

enc = OneHotEncoder()
for feature in cat_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    base_train_csr = sparse.hstack(
    (base_train_csr, enc.transform(train[feature].values.reshape(-1, 1))), 'csr','bool')
    base_predict_csr = sparse.hstack(
    (base_predict_csr, enc.transform(test[feature].values.reshape(-1, 1))),'csr','bool')
print('one-hot prepared !')


cv = CountVectorizer(min_df=10)
for feature in ['new_con']:
    data[feature] = data[feature].astype(str)
    cv.fit(data[feature])
    base_train_csr = sparse.hstack((base_train_csr, cv.transform(train[feature].astype(str))), 'csr', 'bool')
    base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(test[feature].astype(str))), 'csr',
                                     'bool')
print('cv prepared !')


train_csr = sparse.hstack(
    (sparse.csr_matrix(train[num_feature]), base_train_csr), 'csr').astype(
    'float32')
predict_csr = sparse.hstack(
    (sparse.csr_matrix(test[num_feature]), base_predict_csr), 'csr').astype('float32')



def lgb_f1(labels, preds):
    score = f1_score(labels, np.round(preds))
    return 'f1', score, True

lgb_model = lgb.LGBMClassifier(random_seed=2019, n_jobs=-1, objective='binary',
                     learning_rate=0.1, n_estimators=500, num_leaves=100, max_depth=-1,
                     min_child_samples=20, min_child_weight=9, subsample_freq=1,
                     subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=5)


# =============================================================================
# lgb_model = lgb.LGBMClassifier(
#     boosting_type='gbdt', num_leaves=61, reg_alpha=3, reg_lambda=1,
#     max_depth=-1, n_estimators=5000, objective='binary',
#     subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
#     learning_rate=0.035, random_state=2018, n_jobs=10
# )
# =============================================================================
skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
best_score = []
for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
    lgb_model.fit(train_csr[train_index], train_y[train_index],
                  eval_set=[(train_csr[train_index], train_y[train_index]),
                            (train_csr[test_index], train_y[test_index])],
                            eval_metric=lgb_f1,early_stopping_rounds=200, verbose=10)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
    predict_result['label'] = predict_result['label'] + test_pred
predict_result['label'] = predict_result['label'] / 5
mean = predict_result['label'].mean()
predict_result['label']=predict_result['label'].apply(lambda x: 1 if x>=0.5 else 0) ##∪概率大于0.5的置1，否则置0
print('test pre_label distribution:\n',predict_result['label'].value_counts()) ## 模型预测测试集的标签分布
predict_result.to_csv('submit081202.csv',index=None) ##保存为submit0704.csv文件





X = train.values
y = label.values
X_test = test.values

# =============================================================================
# def featureSelect(x_train,y_train,x_val,func=chi2,percentile=80):
#     model = SelectPercentile(func,percentile=percentile)
#     model.fit(x_train,y_train)
#     x_train = model.transform(x_train)
#     x_val = model.transform(x_val)
#     return x_train,x_val
# X,X_test = featureSelect(X,y,X_test)
# 
# =============================================================================

##模型训练预测：
#oof_lgb,prediction_lgb,feature_importance_df = lgb_model(train, label, test)

def f1_score_vail(pred, data_vail):
    labels = data_vail.get_label()
   # pred = np.argmax(pred.reshape(2, -1), axis=0)      # lgb的predict输出为各类型概率值
    temp = pd.DataFrame()
    temp['pred'] = pred
    temp['pred'] = temp['pred'].apply(lambda x: 1 if x>=0.5 else 0)
    score_vail = f1_score(y_true=labels, y_pred=temp['pred'].values, average='macro')
    return 'f1_score', score_vail, True



lgb_params = {
    "learning_rate": 0.05,
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "max_depth": -1,
    "num_leaves": 120,
    "objective": "binary",
    "verbose": -1,
    'feature_fraction': 0.8,
    "min_split_gain": 0.1,
    "boosting_type": "gbdt",
    "subsample": 0.8,
    "min_data_in_leaf": 50,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "tree_method": 'exact',
   # "seed":2019
}

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
oof_lgb=np.zeros(train.shape[0]) ##用于存放训练集概率，由每折验证集所得
prediction_lgb=np.zeros(test.shape[0])  ##用于存放测试集概率，k折最后要除以k取平均
feature_importance_df = pd.DataFrame() ##存放特征重要性，此处不考虑
for index,(train_index, test_index) in enumerate(skf.split(X, y)):
    print('fold:',index+1,'training')
    X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index] 
    train_data = lgb.Dataset(X_train, label=y_train,)
    #feature_name=train.columns.tolist(), categorical_feature=categorical_features)    # 训练数据
    validation_data = lgb.Dataset(X_valid, label=y_valid,)
    #feature_name=train.columns.tolist(), categorical_feature=categorical_features)   # 验证数据
    ##训练：
    clf = lgb.train(lgb_params, train_data, num_boost_round=1500, valid_sets=[validation_data],
                   # categorical_feature=list(range(0, 15)),
                    early_stopping_rounds=100, feval=f1_score_vail, verbose_eval=100)     # 训练
    
    ##预测验证集：
    oof_lgb[test_index] += clf.predict(X_valid, num_iteration=clf.best_iteration)
    ##预测测试集：
    prediction_lgb += clf.predict(X_test, num_iteration=clf.best_iteration)  # 预测
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = list(train.columns)
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = index + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    

print('the roc_auc_score for train:',f1_score(y,np.round(oof_lgb))) ##线下auc评分0.9340750799421788
#0.9356376887472336
#0.9345239957696856
#the roc_auc_score for train: 0.9343666317898881

#the roc_auc_score for train: 0.9334464206912307
#the roc_auc_score for train: 0.9342165777269319
#the roc_auc_score for train: 0.9345978841166886
#the roc_auc_score for train: 0.9344661586685651
#the roc_auc_score for train: 0.9345344631212686
#the roc_auc_score for train: 0.9346680298793975
prediction_lgb/=5
feature_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)

feature_importance.to_csv('importance081101.csv', index=True)


##保存结果：
sub = pd.DataFrame()
sub['sid'] = test_id
sub['label']=prediction_lgb
sub['label']=sub['label'].apply(lambda x: 1 if x>=0.5 else 0) ##∪概率大于0.5的置1，否则置0
print('test pre_label distribution:\n',sub['label'].value_counts()) ## 模型预测测试集的标签分布
sub.to_csv('submit081101.csv',index=None) ##保存为submit0704.csv文件


