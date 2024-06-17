import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_2 = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model_1 = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
sentiment_model = pipeline(model="mo27harakani/finetuning-sentiment-model-3000-samples-2")

# Map raw predictions to languages
id2lang = model_2.config.id2label
lang_for_TTT_model=['jpn','nld','arz','plo','deu','ita','por','tur','spa','hin','ell','urd','bul','eng','fra','cmn','rus','tha','sw','vie']
for i in range(len(id2lang)):
  id2lang[i]=lang_for_TTT_model[i]

def get_rating(pred_rating):
  rating=0
  if pred_rating[0]['label']=='LABEL_0' and (pred_rating[0]['score']>=0.5 and pred_rating[0]['score']<0.7):
    rating=3
  elif pred_rating[0]['label']=='LABEL_0' and (pred_rating[0]['score']>=0.7 and pred_rating[0]['score']<0.8):
    rating=2
  elif pred_rating[0]['label']=='LABEL_0' and (pred_rating[0]['score']>=0.8):
    rating=1
  elif pred_rating[0]['label']=='LABEL_1'and (pred_rating[0]['score']>=0.5 and pred_rating[0]['score']<0.7):
    rating=3
  elif pred_rating[0]['label']=='LABEL_1' and (pred_rating[0]['score']>=0.7 and pred_rating[0]['score']<0.8):
    rating=4
  else:
    rating=5
  return rating

def predict_sentiment(text):
  li_text=[]
  inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
  with torch.no_grad():
    logits = model_2(**inputs).logits

  preds = torch.softmax(logits, dim=-1)
  vals, idxs = torch.max(preds, dim=1)
  source_lang=id2lang[idxs.item()]
  targ_lang='eng'
  # from text
  if source_lang!='eng':
    text_inputs = processor(text =text[0],src_lang=source_lang, return_tensors="pt").to(device)
    model_1.to(device)
    text_array_from_text = model_1.generate(**text_inputs, tgt_lang="eng",generate_speech=False)[0].cpu().numpy().squeeze()
    translated_text_from_text = processor.decode(text_array_from_text.tolist(), skip_special_tokens=True)
    li_text.append(translated_text_from_text)
  else:
    li_text.append(text[0])
  pred_rating=sentiment_model(li_text)
  # print(pred_rating)
  rating=get_rating(pred_rating)

  li_text=[]

  return rating