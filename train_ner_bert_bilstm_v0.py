# -*- coding: utf-8 -*-
"""
Created on Oct 10 10:33:42 2023

@author: Administrator

"""
import os
import sys 
sys.path.append("/workspace/ZoeGPT")
print(sys.path)


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

RANDOM_STATE = 0
LEARNING_RATE = 3e-3
NUM_EPOCHS = 10
batch_size = 64
MAX_LEN = 256
device = 'cuda:1'
# train-val
TEST_SIZE = 0.1 # 9:1 5:5 3:7 2:8 1:9
DATASET_SPLIT = str(f'{(1-TEST_SIZE)*10:.1f}') + ':' + str(TEST_SIZE*10)


# config_path = '/workspace/ZoeGPT/MODELS/aliendao/dataroot/models/bert-base-chinese/config.json'
# checkpoint_path = '/workspace/ZoeGPT/MODELS/bert-base-chinese/tf_v1'
# dict_path = '/workspace/ZoeGPT/MODELS/aliendao/dataroot/models/bert-base-chinese/vocab.txt'
PRETRAINED_MODEL_NAME = '/workspace/ZoeGPT/MODELS/bert-base-chinese'
CORPUS_PATH = '/workspace/ZoeGPT/BISAI/[DataFountain]基于通用大模型的知识库问答/Data/train.json'

SAVE_PATH_FILE = f'results/{os.path.basename(CORPUS_PATH).split(".")[-2]}/BERT-NER/batch-{batch_size}-epoch-{NUM_EPOCHS}-{DATASET_SPLIT}-{LEARNING_RATE}/'
mkdir(SAVE_PATH_FILE)       
src_df = pd.read_json(CORPUS_PATH)
src_df['entity']=None
for idx,row in src_df.iterrows():
    src_df['question'][idx]=[src_df['question'][idx]]
    src_df['entity'][idx]=list(set([triple.split(' ||| ')[0] for triple in row['attribute']]))



# 加载预训练的BERT模型和分词器
bert_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

def find_lcsubstr(s1, s2): 
    m=[[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)] #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0  #最长匹配的长度
    p=0 #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
            if m[i+1][j+1]>mmax:
                mmax=m[i+1][j+1]
                p=i+1
    return s1[p-mmax:p]


class NERDataset(Dataset):
    def __init__(self, datas, entitys, tokenizer, max_length):
        self.datas = datas
        self.entitys = entitys
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        question = self.datas[idx][0]
        entity = self.entitys[idx][0]
        
        # y = [[0] for j in range(MAX_LEN)]
        y = [0]*MAX_LEN
        if entity!='没有找到该问题对应的知识':    
            #得到实体名和问题的最长连续公共子串
            public_substr = find_lcsubstr(question, entity)
            if public_substr in question:
                begin = question.index(public_substr)+1
                end = begin + len(public_substr)
                if end < MAX_LEN-1:
                    for pos in range(begin,end):
                        y[pos] = 1
        
        
        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            # 'question':question,
            # 'entity':entity,
            'idx':idx,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(y, dtype=torch.long)
        }

# 创建一个TensorDataset对象，用于存储训练集的输入特征和标签
train_dataset = NERDataset(src_df['question'], src_df['entity'],tokenizer=tokenizer, max_length=MAX_LEN)

# train-val
train_data, val_data = train_test_split(train_dataset, random_state=RANDOM_STATE, test_size=TEST_SIZE)

def collate_fn(batch):
    # 将batch分成多个列表，每个列表对应一个键值
    question, entity, input_ids, attention_mask, labels = zip(*(batch[0].values()))
    # 对每个列表进行相应的处理，例如堆叠张量，转换类型等
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)
    # 返回一个字典，包含处理后的数据
    return {
        'question': question,
        'entity': entity,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


# 创建一个DataLoader对象，用于将训练集分批次地提供给模型
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

#搭建模型
# 设置BERT模型的所有参数为可训练
for param in bert_model.parameters():
    param.requires_grad = True
# 创建一个基于BERT和双向LSTM的NER模型类
class NERbyBERTBiLSTMSoftmax(nn.Module):
    def __init__(self):
        super(NERbyBERTBiLSTMSoftmax, self).__init__()
        self.num_labels = 1
        self.bert_model = bert_model
        self.bilstm = nn.LSTM(bert_model.config.hidden_size, 256, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(512, self.num_labels)# num_labels=1
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, labels=None):
        # 传入BERT模型，得到输出层
        x = self.bert_model(input_ids, token_type_ids)[0]
        # 传入双向LSTM层，得到输出层
        x, _ = self.bilstm(x)
        # 使用dropout层
        x = self.dropout(x)
        # 传入全连接层，得到输出层
        logits = self.linear(x)
        # 使用sigmoid函数，得到分类概率
        logits = self.sigmoid(logits)
        
        loss = None
        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float()) # labels不需要require_grad=True,损失函数会自动修改
        
        return {
            "loss":loss,
            "logits":logits
        }
# 创建一个模型实例
model = NERbyBERTBiLSTMSoftmax()
model = model.to(device)
# 打印模型的结构和参数信息
# print(model)

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

#训练模型
maxf = 0.0
def computeF(gold_entity,pre_entity):
    '''
    根据标注的实体位置和预测的实体位置，计算prf,完全匹配
    输入： Python-list  3D，值为每个实体的起始位置列表[begin，end]
    输出： float
    '''    
    truenum = 0
    prenum = 0
    goldnum = 0
    for i in range(len(gold_entity)):
        goldnum += len(gold_entity[i])
        prenum  += len(pre_entity[i])
        truenum += len(set(gold_entity[i]).intersection(set(pre_entity[i])))
    try:
        precise = float(truenum) / float(prenum)
        recall = float(truenum) / float(goldnum)
        f = float(2 * precise * recall /( precise + recall)) 
    except:
        precise = recall = f = 0.0
    print('本轮实体的F值是 %f' %(f))
    return precise,recall,f

def restore_entity_from_labels_on_corpus(predicty,questions):
    def restore_entity_from_labels(labels,question):
        entitys = []
        str = ''
        labels = labels[1:-1]
        for i in range(min(len(labels),len(question))):
            if labels[i]==1:
                str += question[i]
            else:
                if len(str):
                    entitys.append(str)
                    str = ''
        if len(str):
            entitys.append(str) 
        return entitys
    all_entitys = []
    for i in range(len(predicty)):
        all_entitys.append(restore_entity_from_labels(predicty[i],questions[i]))
    return all_entitys


for i in range(NUM_EPOCHS):
    # 设置模型为训练模式
    model.train()
    for batch in tqdm(train_loader,leave=False):
        # 将数据移动到设备上，例如GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播，得到输出层
        outputs = model(input_ids, attention_mask, labels)
        # 计算损失函数
        loss = outputs["loss"]
        # 反向传播，计算梯度
        loss.backward()
        # 更新模型的参数
        optimizer.step()
        scheduler.step()
        
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            total_loss += loss.item()   
             
            logits = outputs["logits"].tolist()#(num,len,1)
            predicty = [[1 if each[0]>0.01 else 0 for each in line] for line in logits]
            questions = [src_df['question'][data_idx.item()] for data_idx in batch['idx']]
            predict_entitys = restore_entity_from_labels_on_corpus(predicty,questions)
            
            entitys = [src_df['entity'][data_idx.item()] for data_idx in batch['idx']]
            p,r,f = computeF(entitys,predict_entitys)
        print ('%d epoch f-score is %.3f'%(i,f))
        if f>maxf:
            # model.save_pretrained(SAVE_PATH_FILE)
            # 或者保存整个模型对象
            torch.save(model, SAVE_PATH_FILE+'ner_model.pt')

            maxf = f
        
# model = torch.load(SAVE_PATH_FILE+'ner_model.pt')
# model.eval()
# predicty = model([testx1,testx2],batch_size=32).tolist()#(num,len,1)
# predicty = [[1 if each[0]>0.5 else 0 for each in line] for line in predicty]
# predict_entitys = restore_entity_from_labels_on_corpus(predicty,test_questions)
# for j in range(300,320):
#     print (predict_entitys[j])
#     print (test_entitys[j])
