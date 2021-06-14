### 电商用户购买行为预测

#### 比赛介绍

互联网的出现和普及给用户带来了大量的信息，满足了用户在信息时代对信息的需求，但是网上信息量的大幅增长也带来了“信息过载”的问题。这使得用户在面对大量信息时无法从中获得对自己真正有用的信息，导致用户对信息的使用效率大大降低了。为了帮助用户更快速地过滤出有用的信息，需要依据真实的用户购买行为记录，利用机器学习相关技术建立稳健的电商用户购买行为预测模型。

#### 意义

1. 用于预测用户的下一个行为，以此为用户进行商品的推荐，
2. 准确捕获用户的购买兴趣，提高电商平台商品的购买率
3. 提升购物体验，促进电子商务发展

#### 数据集

先查看大赛提供的train.csv训练数据集：
数据集的字段的含义：

<img
src=https://raw.githubusercontent.com/youminglan/Picture/main/img/20210518233004.png>

![image-20210518233010385](https://img-blog.csdnimg.cn/img_convert/b52cd6bda79878237bf38afa70e2b23b.png)

#### 模型选择

回归预测问题
可供选择：
1. 机器学习模型：逻辑回归，支持向量机SVM，决策树，融合逻辑回归和支持向量机

2. 深度学习模型：CNN-LSTM

使用CNN从用户行为中提取高影响力的特征，然后利用LSTM建立时间序列预测模型，最后通过全连接层输出模型预测结果

#### 数据预处理

首先进行数据清洗：数据处理和特征构建完成用户历史行为数据清洗，剔除刷单用户、重大促销等不具有一般规律的数据，并采用分段下采样方法进行样本均衡处理。

代码实现：

```python
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

raw_train = pd.read_csv('./data/train.csv')
raw_test = pd.read_csv('./data/test.csv')
submit_df = pd.read_csv('./data/submit_example.csv')

display(raw_train, raw_test, submit_df)

#预处理
for df in [raw_train, raw_test]:
    # 处理空值
    for f in ['category_code', 'brand']:
        df[f].fillna('<unkown>', inplace=True)

    # 处理时间
    df['event_time'] = pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S UTC')
    df['timestamp'] = df['event_time'].apply(lambda x: time.mktime(x.timetuple()))
    df['timestamp'] = df['timestamp'].astype(int)
    
# 排序
raw_train = raw_train.sort_values(['user_id', 'timestamp'])
raw_test = raw_test.sort_values(['user_id', 'timestamp'])

# 处理非数值特征
df = pd.concat([raw_train, raw_test], ignore_index=True)

for f in ['event_type', 'category_code', 'brand']:
    # 构建编码器
    le = LabelEncoder()
    le.fit(df[f])

    # 设置新值
    raw_train[f] = le.transform(raw_train[f])
    raw_test[f] = le.transform(raw_test[f])
    
# 删除无用列
useless = ['event_time', 'user_session', 'timestamp']
for df in [raw_train, raw_test]:
    df.drop(columns=useless, inplace=True)
#滑动窗口构造数据集
#为了让机器学习模型能够处理时序数据，必须通过滑动窗口构造数据，后一个时间点的作为前一个时间点的预测值

# 训练集数据生成：滑动窗口
# 用前一个时间节点的数据预测后一个时间节点是商品
train_df = pd.DataFrame()
user_ids = raw_train['user_id'].unique()
for uid in tqdm(user_ids):
    user_data = raw_train[raw_train['user_id'] == uid].copy(deep=True)
    if user_data.shape[0] < 2:
        # 小于两条的，直接忽略
        continue

    user_data['y'] = user_data['product_id'].shift(-1)
    user_data = user_data.head(user_data.shape[0]-1)
    train_df = train_df.append(user_data)

train_df['y'] = train_df['y'].astype(int)
train_df = train_df.reset_index(drop=True)

# 测试集数据生成，只取每个用户最后一次操作用来做预测
test_df = raw_test.groupby(['user_id'], as_index=False).last()

train_df.drop(columns=['user_id'], inplace=True)

display(train_df, test_df)

user_ids = test_df['user_id'].unique()

preds = []
for uid in tqdm(user_ids):
    pids = raw_test[raw_test['user_id'] == uid]['product_id'].unique()

    # 找到训练集中有这些product_id的数据作为当前用户的训练集
    p_train = train_df[train_df['product_id'].isin(pids)]
    
    # 只取最后一条进行预测
    user_test = test_df[test_df['user_id'] == uid].drop(columns=['user_id'])

    X_train = p_train.iloc[:, :-1]
    y_train = p_train['y']

    if len(X_train) > 0:
        # 训练
        clf = lgb.LGBMClassifier(**{'seed': int(time.time())})
        clf.fit(X_train, y_train)
    
        # 预测
        pred = clf.predict(user_test)[0]
    else:
        # 训练集中无对应数据
        # 直接取最后一条数据作为预测值
        pred = user_test['product_id'].iloc[0]

    preds.append(pred)

submit_df['product_id'] = preds

# 分数 0.206
submit_df.to_csv('baseline.csv', index=False)


```