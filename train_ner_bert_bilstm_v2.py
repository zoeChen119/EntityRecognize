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
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, AdamW
# from transformers.models.bert import convert_bert_pytorch_checkpoint_to_original_tf 
# python3 -m transformers.models.bert.convert_bert_pytorch_checkpoint_to_original_tf --model_name /workspace/ZoeGPT/MODELS/bert-base-chinese --pytorch_model_path /workspace/ZoeGPT/MODELS/bert-base-chinese/pytorch_model.bin --tf_cache_dir /workspace/ZoeGPT/MODELS/bert-base-chinese/tf

RANDOM_STATE = 0
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
batch_size = 32
MAX_LEN = 512
device = 'cuda:1'
# train-val
TEST_SIZE = 0.1 # 9:1 5:5 3:7 2:8 1:9
DATASET_SPLIT = str(f'{(1-TEST_SIZE)*10:.1f}') + ':' + str(TEST_SIZE*10)

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
        self.num_labels = 3
        self.bert_model = bert_model
        self.bilstm = nn.LSTM(bert_model.config.hidden_size, 256, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(512, self.num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.bert_model(input_ids, attention_mask)[0]
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        # 使用sigmoid函数，得到分类概率
        logits = self.sigmoid(logits)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 只计算非填充部分的损失
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        
        return {
            "loss":loss,
            "logits":logits
        }

# 创建一个模型实例
model = NERbyBERTBiLSTMSoftmax()
model = model.to(device)
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
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader,leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        
    model.eval()
    valid_loss = 0
    valid_acc = 0
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).sum() / len(attention_mask.view(-1))
            # # predicty = [[1 if each[0]>0.5 else 0 for each in line] for line in logits]
            # predicty = list(torch.argmax(probs, dim=-1))
            # questions = [src_df['question'][data_idx.item()] for data_idx in batch['idx']]
            # predict_entitys = restore_entity_from_labels_on_corpus(predicty,questions)
            
            # entitys = [src_df['entity'][data_idx.item()] for data_idx in batch['idx']]
            # p,r,f = computeF(entitys,predict_entitys)
        valid_loss += loss.item() 
        valid_acc += acc.item()
        
    scheduler.step()
    
    # 打印训练和验证的结果
    print(f'Epoch {i + 1}:')
    print(f'Train loss: {train_loss / len(train_loader):.4f}')
    print(f'Valid loss: {valid_loss / len(val_loader):.4f}')
    print(f'Valid acc: {valid_acc / len(val_loader):.4f}')
    # print ('%d epoch f-score is %.3f'%(i,f))
    # if f>maxf:
    #     # model.save_pretrained(SAVE_PATH_FILE)
    #     # 或者保存整个模型对象
    #     torch.save(model, SAVE_PATH_FILE+'ner_model.pt')

    #     maxf = f

# model = torch.load(SAVE_PATH_FILE+'ner_model.pt')
# model.eval()
# predicty = model([testx1,testx2],batch_size=32).tolist()#(num,len,1)
# predicty = [[1 if each[0]>0.5 else 0 for each in line] for line in predicty]
# predict_entitys = restore_entity_from_labels_on_corpus(predicty,test_questions)
# for j in range(300,320):
#     print (predict_entitys[j])
#     print (test_entitys[j])
