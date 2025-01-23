import torch
import torch.nn as nn
import numpy as np

from transformers import DistilBertTokenizer, DistilBertModel
# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Import the DistilBert pretrained model
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

DEVICE = 'cpu'
MODEL_DICT_PATH = './data.pth'

model_dict = torch.load(MODEL_DICT_PATH, map_location=torch.device('cpu'))
# print(model_dict.keys())

NUM_CLASSES = model_dict['output_size']


class BERT_Arch(nn.Module):
   def __init__(self, bert):
       super(BERT_Arch, self).__init__()
       self.bert = bert

       # dropout layer
       self.dropout = nn.Dropout(0.2)

       # relu activation function
       self.relu =  nn.ReLU()
       # dense layer
       self.fc1 = nn.Linear(768,512)
       self.fc2 = nn.Linear(512,256)
       self.fc3 = nn.Linear(256,NUM_CLASSES)
       #softmax activation function
       self.softmax = nn.LogSoftmax(dim=1)
       #define the forward pass
   def forward(self, sent_id, mask):
      #pass the inputs to the model
      cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]

      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)

      x = self.fc2(x)
      x = self.relu(x)
      x = self.dropout(x)
      # output layer
      x = self.fc3(x)

      # apply softmax activation
      x = self.softmax(x)
      return x

CHATBOT_MODEL = BERT_Arch(bert)
CHATBOT_MODEL.load_state_dict(model_dict['model_state'])
