# -*- coding: utf-8 -*-
"""
Created on Oct 10 10:33:42 2023

@author: Administrator

"""
import os
import sys 
sys.path.append("/workspace/ZoeGPT")
print(sys.path)
import re
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm
import os
import time
import json
from transformers import BertTokenizerFast
from transformers import logging
import pandas as pd
from tqdm import tqdm
from utils_zoe import mkdir
import torch
from torch import nn
from torch.nn import BCELoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, AdamW
# from transformers.models.bert import convert_bert_pytorch_checkpoint_to_original_tf 
# python3 -m transformers.models.bert.convert_bert_pytorch_checkpoint_to_original_tf --model_name /workspace/ZoeGPT/MODELS/bert-base-chinese --pytorch_model_path /workspace/ZoeGPT/MODELS/bert-base-chinese/pytorch_model.bin --tf_cache_dir /workspace/ZoeGPT/MODELS/bert-base-chinese/tf


device = 'cuda:3'

# config_path = '/workspace/ZoeGPT/MODELS/aliendao/dataroot/models/bert-base-chinese/config.json'
# checkpoint_path = '/workspace/ZoeGPT/MODELS/bert-base-chinese/tf_v1'
# dict_path = '/workspace/ZoeGPT/MODELS/aliendao/dataroot/models/bert-base-chinese/vocab.txt'
# PRETRAINED_MODEL_NAME = '/workspace/ZoeGPT/MODELS/bert-base-chinese'
CORPUS_PATH = '/workspace/ZoeGPT/BISAI/[DataFountain]基于通用大模型的知识库问答/Data/train.json'
   
src_df = pd.read_json(CORPUS_PATH)
src_df['entity']=None
for idx,row in src_df.iterrows():
    src_df['question'][idx]=[src_df['question'][idx]]
    src_df['entity'][idx]=list(set([triple.split(' ||| ')[0] for triple in row['attribute']]))

src_df['text'], src_df['label'] = '', ''
for i in range(len(src_df)):
    src_df['text'].iloc[i] = src_df['question'].iloc[i][0]
    src_df['label'].iloc[i] = src_df['entity'].iloc[i][0]

src_df = src_df.query('label != "没有找到该问题对应的知识"')
texts, labels = src_df['text'].tolist(), src_df['label'].tolist()

# print(texts)
# print(labels)

def text_bio(texts):
    """
    文本先用 O 标注
    """
    texts_tag = []
    for text in texts:
        tag_row = ''
        for i in range(len(text)):
            tag_row += 'O '
        tag_row = tag_row.rstrip(" ")
        texts_tag.append(tag_row.split(' '))

    return texts_tag

    
def labels_bio(labels):
    """
    标签 BIO 标注
    """
    labels_tag = []
    for label in labels:
        tag_row = ''
        for length in range(len(label)):
            if length == 0:
                tag_row += 'B-E '
            else:
                tag_row += 'I-E '
        tag_row = tag_row.rstrip(" ")
        labels_tag.append(tag_row.split(' '))

    return labels_tag


def tag_map(texts, texts_tag, labels, labels_tag):
    """
    标签 BIO 标注的结果去替换文本对应位置的 O
    """
    texts_tag_new = []
    for i in range(len(texts_tag)):
        text_tag_raw = texts_tag[i]
        label_tag_raw = labels_tag[i]

        start_idx = texts[i].find(labels[i])
        end_idx = start_idx + len(labels[i])
        
        text_tag_raw[start_idx: end_idx] = label_tag_raw

        # text_tag_raw = " ".join(text_tag_raw)
        texts_tag_new.append(text_tag_raw)

    return texts_tag_new


def get_label_list(texts_tag_new):
        """
        用于获取数据集中字符标签的所有类别
        """
        tag_lst = []
        for single_tag in texts_tag_new:  # tag_list: xxx行list标签
            for tag in set(single_tag):  # 元素标签去重
                tag_lst.append(tag)  # 1行list标签
        tag_list = list(set(tag_lst))  # 去重: ['I-B', 'B-E', 'B-B', 'I-E', 'I-P', 'B-P', 'O']
        label_list = sorted(tag_list)  # ['B-B', 'B-E', 'B-P', 'I-B', 'I-E', 'I-P', 'O']

        return label_list


def generate_dataset(texts, texts_tag_new):
    all_texts = []
    for text in texts:
        all_texts.append({'text': list(text)})

    text_labels = [ ]
    for label in texts_tag_new:
        text_labels.append({'label': label})
        
    text = pd.DataFrame(all_texts)  # 文本
    label = pd.DataFrame(text_labels)  # 标签
    dataset = pd.concat([text, label], axis=1)  # 合并

    return dataset


def dataset_split(dataset):
    """
    数据集划分：dataset -> train.txt、dev.txt、test.txt
    """
    # 分割数据集 6:2:2
    dataset['index'] = dataset.index
    train = dataset.sample(frac=0.6, random_state=42)
    df_rest = dataset[~dataset.index.isin(train['index'])]
    dev = df_rest.sample(frac=0.5, random_state=42)
    test = df_rest[~df_rest.index.isin(dev['index'])]

    # words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
    # labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']

    train_data = []
    for i in range(len(train)):
        train_data.append((train['text'].iloc[i], train['label'].iloc[i]))

    dev_data = []
    for i in range(len(dev)):
        dev_data.append((dev['text'].iloc[i], dev['label'].iloc[i]))

    test_data = []
    for i in range(len(test)):
        test_data.append((test['text'].iloc[i], test['label'].iloc[i]))

    return train_data, dev_data, test_data 


texts_tag = text_bio(texts)
labels_tag = labels_bio(labels)
texts_tag_new = tag_map(texts, texts_tag, labels, labels_tag)
label_list = get_label_list(texts_tag_new)
print(label_list)
dataset = generate_dataset(texts, texts_tag_new)
train_data, dev_data, test_data = dataset_split(dataset)
print()



# 加载预训练的BERT模型和分词器
# bert_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)

# tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# 设置transformers模块的日志等级，减少不必要的警告，对训练过程无影响，请忽略
logging.set_verbosity_error()

# 环境变量：设置程序能使用的GPU序号。例如：
# 当前服务器有8张GPU可用，想用其中的第2、5、8卡，这里应该设置为:
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# 通过继承nn.Module类自定义符合自己需求的模型
class BertNERModel(nn.Module):

    # 初始化类
    def __init__(self, ner_labels, pretrained_name='/workspace/ZoeGPT/MODELS/bert-base-chinese'):
        """
        Args: 
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        super(BertNERModel, self).__init__()
        # 加载HuggingFace的BertModel
        # BertModel的最终输出维度默认为768
        # return_dict=True 可以使BertModel的输出具有dict属性，即以 bert_output['last_hidden_state'] 方式调用
        self.bert = BertModel.from_pretrained(pretrained_name)
        # 通过一个线性层将标签对应的维度：768->class_size
        self.classifier = nn.Linear(768, ner_labels)

    def forward(self, inputs):
        # 获取DataLoader中已经处理好的输入数据：
        # input_ids :tensor类型，shape=batch_size*max_len   max_len为当前batch中的最大句长
        # input_tyi :tensor类型，
        # input_attn_mask :tensor类型，因为input_ids中存在大量[Pad]填充，attention mask将pad部分值置为0，让模型只关注非pad部分
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
            'token_type_ids'], inputs['attention_mask']
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        # bert_output 分为两个部分：
        #   last_hidden_state:最后一个隐层的值
        #   pooler output:对应的是[CLS]的输出,用于分类任务
        # categories_numberic：tensor类型，shape=batch_size*class_size，用于后续的CrossEntropy计算
        categories_numberic = self.classifier(output.last_hidden_state)
        batch_size, seq_len, ner_class_num = categories_numberic.shape
        categories_numberic = categories_numberic.view(
            (batch_size * seq_len, ner_class_num))
        return categories_numberic


# def save_pretrained(model, path):
#     # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
#     os.makedirs(path, exist_ok=True)
#     torch.save(model, os.path.join(path, 'model.pth'))


class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # 这里可以自行定义，Dataloader会使用__getitem__(self, index)获取数据
        # 这里我设置 self.dataset[index] 规定了数据是按序号取得，序号是多少DataLoader自己算，用户不用操心
        return self.dataset[index]


def coffate_fn(examples):
    # print(examples)
    sents, all_labels = [], []
    for example in examples:
        # for sent, ner_labels in examples[i]:
        sents.append(example[0])
        all_labels.append([categories[label] for label in example[1]])
    tokenized_inputs = tokenizer(sents,
                                 truncation=True,
                                 padding=True,
                                 return_offsets_mapping=True,
                                 is_split_into_words=True,
                                 max_length=512,
                                 return_tensors="pt")
    targets = []
    for i, labels in enumerate(all_labels):
        label_ids = []
        for word_idx in tokenized_inputs.word_ids(batch_index=i):
            # 将特殊符号的标签设置为-100，以便在计算损失函数时自动忽略
            if word_idx is None:
                label_ids.append(-100)
            else:
                # 把标签设置到每个词的第一个token上
                label_ids.append(labels[word_idx])
        targets.append(label_ids)
    targets = torch.tensor(targets)
    return tokenized_inputs, targets


def split_entity(label_sequence):
    entity_mark = dict()
    entity_pointer = None
    for index, label in enumerate(label_sequence):
        if label.startswith('B'):
            category = label.split('-')[1]
            entity_pointer = (index, category)
            entity_mark.setdefault(entity_pointer, [label])
        elif label.startswith('I'):
            if entity_pointer is None:
                continue
            if entity_pointer[1] != label.split('-')[1]:
                continue
            entity_mark[entity_pointer].append(label)
        else:
            entity_pointer = None
    return entity_mark


def evaluate(real_label, predict_label):
    # 序列标注的准确率和召回率计算，详情查看：https://zhuanlan.zhihu.com/p/56582082
    real_entity_mark = split_entity(real_label)
    predict_entity_mark = split_entity(predict_label)

    true_entity_mark = dict()
    key_set = real_entity_mark.keys() & predict_entity_mark.keys()
    for key in key_set:
        real_entity = real_entity_mark.get(key)
        predict_entity = predict_entity_mark.get(key)
        if tuple(real_entity) == tuple(predict_entity):
            true_entity_mark.setdefault(key, real_entity)

    real_entity_num = len(real_entity_mark)
    predict_entity_num = len(predict_entity_mark)
    true_entity_num = len(true_entity_mark)

    precision = true_entity_num / predict_entity_num
    recall = true_entity_num / real_entity_num
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


# 训练准备阶段，设置超参数和全局变量

batch_size = 32
num_epoch = 5  # 训练轮次
learning_rate = 5e-5  # 优化器的学习率

# 获取训练、测试数据、分类类别总数
# train_data, test_data = load_sentence_nertags(data_path=data_path,
#                                               train_ratio=train_ratio)
# ['B-E', 'I-E', 'O']
categories = {
    'O': 0,
    'B-E': 1,
    'I-E': 2,
    0: 'O',
    1: 'B-E',
    2: 'I-E'
}

# 将训练数据和测试数据的列表封装成Dataset以供DataLoader加载
train_dataset = BertDataset(train_data)
test_dataset = BertDataset(test_data)
"""
DataLoader主要有以下几个参数：
Args:
    dataset (Dataset): dataset from which to load the data.
    batch_size (int, optional): how many samples per batch to load(default: ``1``).
    shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
    collate_fn : 传入一个处理数据的回调函数
DataLoader工作流程：
1. 先从dataset中取出batch_size个数据
2. 对每个batch，执行collate_fn传入的函数以改变成为适合模型的输入
3. 下个epoch取数据前先对当前的数据集进行shuffle，以防模型学会数据的顺序而导致过拟合
"""
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=coffate_fn,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn,
                             shuffle=True)

#固定写法，可以牢记，cuda代表Gpu
# torch.cuda.is_available()可以查看当前Gpu是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型，因为这里是英文数据集，需要用在英文上的预训练模型：bert-base-uncased
# uncased指该预训练模型对应的词表不区分字母的大小写
# 详情可了解：https://huggingface.co/bert-base-uncased
pretrained_model_name = '/workspace/ZoeGPT/MODELS/bert-base-chinese'

model = BertNERModel(3, pretrained_model_name)
# 固定写法，将模型加载到device上，
# 如果是GPU上运行，此时可以观察到GPU的显存增加
model.to(device)
# 加载预训练模型对应的tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

# 训练过程
# Adam是最近较为常用的优化器，详情可查看：https://www.jianshu.com/p/aebcaf8af76e
optimizer = Adam(model.parameters(), learning_rate)  # 使用Adam优化器
CE_loss = nn.CrossEntropyLoss(ignore_index=-100)  # 使用crossentropy作为分类任务的损失函数

# 记录当前训练时间，用以记录日志和存储
timestamp = time.strftime("%m_%d_%H_%M", time.localtime())

# 开始训练，model.train()固定写法，详情可以百度
model.train()
for epoch in range(1, num_epoch + 1):
    # 记录当前epoch的总loss
    total_loss = 0
    # tqdm用以观察训练进度，在console中会打印出进度条

    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
        # tqdm(train_dataloader, desc=f"Training Epoch {epoch}") 会自动执行DataLoader的工作流程，
        # 想要知道内部如何工作可以在debug时将断点打在 coffate_fn 函数内部，查看数据的处理过程

        # 对batch中的每条tensor类型数据，都执行.to(device)，
        # 因为模型和数据要在同一个设备上才能运行
        inputs, targets = [x.to(device) for x in batch]
        targets = targets.view(-1)
        # 清除现有的梯度
        optimizer.zero_grad()

        # 模型前向传播，model(inputs)等同于model.forward(inputs)
        bert_output = model(inputs)

        # 计算损失，交叉熵损失计算可参考：https://zhuanlan.zhihu.com/p/159477597
        loss = CE_loss(bert_output, targets)

        # 梯度反向传播
        loss.backward()

        # 根据反向传播的值更新模型的参数
        optimizer.step()

        # 统计总的损失，.item()方法用于取出tensor中的值
        total_loss += loss.item()

    #测试过程
    target_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Testing"):
            inputs, targets = [x.to(device) for x in batch]
            targets = targets.view(-1)
            bert_output = model(inputs)
            predictions = bert_output.argmax(dim=-1)
            target_labels += [categories[i]
                              for i in targets.tolist() if i != -100]
            pred_labels += [
                categories[i] for i in predictions.tolist()[1:-1] if i != -100
            ]

    precision, recall, f1 = evaluate(real_label=target_labels,
                                     predict_label=pred_labels)
    print("precision is {}\nrecall is {}\nf1 is {}".format(
        precision, recall, f1))

    # if epoch % check_step == 0:
    #     # 保存模型
    #     checkpoints_dirname = "bert_ner_" + timestamp
    #     os.makedirs(checkpoints_dirname, exist_ok=True)
        # save_pretrained(model,
        #                 checkpoints_dirname + '/checkpoints-{}/'.format(epoch))
