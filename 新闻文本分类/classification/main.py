import re
import numpy as np
import jieba

import matplotlib.pyplot as plt


import os

import tensorflow as tf
from tensorflow import keras
train_texts_orgs = []
train_target = []

with open('input/positive_samples.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        dic = eval(line)
        train_texts_orgs.append(dic['text'])
        train_target.append(dic['label'])

with open('input/negative_samples.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        dic = eval(line)
        train_texts_orgs.append(dic['text'])
        train_target.append(dic['label'])

"""数据预处理"""
word_all = set()
for text in train_texts_orgs:
    text = re.sub("[\r|\n|\\s!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·]+", "", text)
    cut = jieba.lcut(text)
    for word in cut:
        word_all.add(word)

print(len(word_all))

word2id = dict()
for i, word in enumerate(word_all):
    word2id[word] = i
# print(word2id)


train_x_all = []
for text in train_texts_orgs:
    text = re.sub("[\r|\n|\\s!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·]+", "", text)

    data = list()
    cut = jieba.lcut(text)
    for word in cut:
        index = word2id[word]
        data.append(index)
    train_x_all.append(data)

print(train_x_all[1])

num_tokens = [ len(tokens) for tokens in train_x_all]
plt.hist(num_tokens, bins=50)
plt.ylabel('number of tokens')
plt.xlabel('length of tokens')
plt.title('Distribution of tokens length')
plt.show()

train_x_all = keras.preprocessing.sequence.pad_sequences(
    train_x_all,
    value=0,
    padding='post', #pre表示在句子前面填充，post表示在句子末尾填充
    maxlen=85
)

train_ds = tf.data.Dataset.from_tensor_slices((train_x_all, train_target)).shuffle(20000).batch(32)

exp_input_batch, exp_target_batch = next(iter(train_ds))
print(exp_input_batch)
print(exp_target_batch)


"""Model"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_all), 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(loss=keras.losses.binary_crossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy']
              )

history = model.fit(train_ds, epochs=10)