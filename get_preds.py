import torch
from get_model import CHATBOT_MODEL, tokenizer
import re
import numpy as np
import json
import random

import warnings
warnings.filterwarnings("ignore")

DEVICE = 'cpu'
MODEL_DICT_PATH = './data.pth'

response_data = json.load(open('./intents_responses.json', 'r'))
response_data

model_dict = torch.load(MODEL_DICT_PATH, map_location=torch.device('cpu'))

NUM_CLASSES = model_dict['output_size']
TAGS = model_dict['tags']
MAX_SEQ_LEN = model_dict['output_size']
INVERSE_LABEL_MAPPING = model_dict['inverse_label_mapping']

# print(INVERSE_LABEL_MAPPING)

def get_prediction(s):
 s = re.sub(r'[^a-zA-Z ]+', '', s)
 test_text = [s]
 CHATBOT_MODEL.eval()

 tokens_test_data = tokenizer(
 test_text,
 max_length = MAX_SEQ_LEN,
 pad_to_max_length=True,
 truncation=True,
 return_token_type_ids=False
 )
 test_seq = torch.tensor(tokens_test_data['input_ids'])
 test_mask = torch.tensor(tokens_test_data['attention_mask'])

 preds = None
 with torch.no_grad():
   preds = CHATBOT_MODEL(test_seq.to(DEVICE), test_mask.to(DEVICE))
#  preds = preds.detach().cpu().numpy()
   preds = preds.detach().cpu().numpy()
 preds = np.argmax(preds, axis = 1)
#  print("Intent Identified: ", INVERSE_LABEL_MAPPING[preds])
 return INVERSE_LABEL_MAPPING[preds[0]]


def get_response(message):
  intent = get_prediction(message)
  for i in response_data['intents']:
    if i["tag"] == intent:
      result = random.choice(i["responses"])
      break
  # print(f"Response : {result}")
  return "Intent: "+ intent + '\n' + "Response: " + result
#   return "AI BOT: " + result