#### 数据简介

数据整理自一家中等化妆品在线商店公布的网上公开数据集，为该化妆品商店真实的用户交易信息，数据集中每一行表示一个事件，所有的事件都与商品和用户相关，并且用户的点击行为之间是有时间顺序的。数据集中包含了商品和用户的多个属性，例如商品编号、商品类别、用户编号、事件时间等。

#### 数据说明

数据文件夹包含3个文件，依次为：

| 文件类别 |     文件名     |                     文件内容                     |
| :------: | :------------: | :----------------------------------------------: |
|  训练集  |   train.csv    | 训练数据集，标签为每个用户序列的最后一个商品编号 |
|  测试集  |    test.csv    |                    测试数据集                    |
| 提交样例 | submission.csv |          仅有两个字段user_id\product_id          |

文件字段说明:

| 名称          | 标签                                                        |
| ------------- | ----------------------------------------------------------- |
| event_time    | When event is was happened                                  |
| event_type    | Event type: one of [view, cart, remove_from_cart, purchase] |
| product_id    | Product ID                                                  |
| category_id   | Product category ID                                         |
| category_code | Category meaningful name (if present)                       |
| brand         | Brand name in lower case (if present)                       |
| price         | Product price                                               |
| user_id       | Permanent user ID                                           |
| user_session  | User session ID                                             |

#### 提交要求

建议提交方式：
参赛者以csv文件格式提交，提交模型结果到大数据竞赛平台，平台进行在线评分，实时排名。目前平台仅支持单文件提交，即所有提交内容需要放在一个文件中；submission.csv文件字段如下：

|   字段名   | 类型 | 取值范围 | 字段解释 |
| :--------: | :--: | :------: | :------: |
|  user_id   | Int  |    -     |  用户ID  |
| product_id | Int  |    -     |  商品ID  |

#### 提交示例

示例如下：

| user_id | product_id |
| :-----: | :--------: |
|  53978  |  5651977   |
|  53980  |  5877766   |

#### 评测标准

本赛题采用召回率和平均倒数排名两个指标进行评价：
![image.png](https://s3.cn-north-1.amazonaws.com.cn/files.datafountain.cn/uploads/admin/editor/2020-11-03/image-607009.png)
其中，TP是真正类，FN是假负类。是商品在推荐列表中的排名Ranki