import pandas as pd

df_main = pd.read_csv('data/packages.csv')
df_main['clickratio'] = df_main['clicks'] / df_main['impressions']
df_chosen = df_main[df_main.clicks >= df_main.clicks.median()].sort_values('clicks', ascending=False)
df_repeats_hl = pd.DataFrame(data=df_chosen.headline + df_chosen.lede)
for col in ['clicks','impressions','clickratio']:
    df_repeats_hl[col] = df_chosen[col]
df_sorted_hl = df_repeats_hl.sort_values('clickratio', ascending=False)
df_hl = df_sorted_hl.rename(columns={0: "headline_lede"}).drop_duplicates(subset=['headline_lede'])


#df_country = pd.read_csv('data/analytics_country_data.csv')
#df_dau = pd.read_csv('data/analytics_daily_users.csv')
#df_dpv = pd.read_csv('data/analytics_daily_pageviews.csv')


# packages_feat = ['created_at', 'test_week', 'test_id', 'headline', 'image_id', 'excerpt',
#        'lede', 'slug', 'share_text', 'share_image', 'impressions', 'clicks',
#        'first_place', 'winner']
# adu_feat = ['date', 'users', 'new_users', 'sessions_per_user', 'sessions',
#        'session_duration', 'bounce_rate', 'pageviews', 'pages_per_session']
# actry_feat = ['country', 'users', 'new_users', 'sessions', 'bounce_rate',
#        'pages_per_session', 'session_duration']
# pgviews = ['pageviews']

import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode some inputs
chosen_entry = df_chosen.loc[0,:]
#text_1 = "Who was Jim Henson ?"
text_1 = chosen_entry.headline
#text_2 = "Jim Henson was a puppeteer"
text_2 = chosen_entry.lede

indexed_tokens_1 = tokenizer.encode(text_1)
indexed_tokens_2 = tokenizer.encode(text_2)

# Convert inputs to PyTorch tensors
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])

# Load pre-trained model (weights)
model = GPT2Model.from_pretrained('gpt2')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor_1 = tokens_tensor_1.to('cuda')
tokens_tensor_2 = tokens_tensor_2.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    hidden_states_1, past = model(tokens_tensor_1)
    # past can be used to reuse precomputed hidden state in a subsequent predictions
    # (see beam-search examples in the run_gpt2.py example).
    hidden_states_2, past = model(tokens_tensor_2, past=past)

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor_1 = tokens_tensor_1.to('cuda')
tokens_tensor_2 = tokens_tensor_2.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    predictions_1, past = model(tokens_tensor_1)
    # past can be used to reuse precomputed hidden state in a subsequent predictions
    # (see beam-search examples in the run_gpt2.py example).
    predictions_2, past = model(tokens_tensor_2, past=past)

# get the predicted last token
predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
predicted_token = tokenizer.decode([predicted_index])


from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np 


model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def score(tokens_tensor):
    loss=model(tokens_tensor, labels=tokens_tensor)[0]
    return np.exp(loss.cpu().detach().numpy())

#texts = [text_1 + text_2 ]
#for text in texts:
text_list = []
perp_list = []
df_hl_1000 = df_hl.iloc[:1000,:].copy()
for row in range(len(df_hl_1000)):
    text = df_hl_1000.iloc[row,0]
    perp_score = tokenizer.encode( text, add_special_tokens=False, return_tensors="pt")        
    perp_list.append(score(perp_score) )  
df_hl_1000['perp_score'] = perp_list

print (text, score(tokens_tensor))

#######
perp_list = []
df_jon = pd.read_csv('data/cleaned_up_ledes.csv')
ct = 0
for row in range(len(df_jon)):
    text = f'{df_jon.headline.iloc[row]} {df_jon.lede.iloc[row]}'
    perp_score = tokenizer.encode( text, add_special_tokens=False, return_tensors="pt")        
    perp_list.append(score(perp_score) )
    ct +=1
    if ct%100 == 0:
        print(f'{ct}/{len(df_jon)}')
df_jon['perp_score'] = perp_list
