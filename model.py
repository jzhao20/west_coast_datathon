from transformers import GPT2LMHeadModel, GPT2Tokenizer
from os.path import exists
import numpy as np 
import pandas as pd
import torch
import math

model= GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if not exists("./training_data/cleaned_up_ledes.csv"):
    df=pd.read_csv("./training_data/packages.csv")
    headlines=df['headline']
    ledes=df['lede']
    clicks=df['clicks']
    impressions=df['impressions']
    winner=df['winner']
    #clean up so there are no duplicates and then do this save it to a csv file that's called cleaned up with headlines and ledes
    set_cleaned_up={}
    for i in range(0,len(ledes)):
        item = (headlines[i],ledes[i])
        if item not in set_cleaned_up or winner[i]:
            set_cleaned_up[item]=[clicks[i],impressions[i],winner[i]]
    headlines=[]
    ledes=[]
    clicks=[]
    impressions=[]
    winner=[]
    click_rate=[]
    for elements in set_cleaned_up.keys():
        if not (pd.isna(elements[0]) or pd.isna(elements[1])):
            headline=elements[0]
            headline="".join(a for a in headline if a not in ("*","(",")","^","#",'"',":",";",">","<"))
            lede=elements[1].replace('<p>','').replace('</p>','')
            lede="".join(u for u in lede if u not in ("*","(",")","^","#",'"',":",";",">","<"))
            lede=lede.strip()
            lede=lede.lstrip(" ")
            headlines.append(headline)
            ledes.append(lede)
            clicks.append(set_cleaned_up[elements][0])
            impressions.append(set_cleaned_up[elements][1])
            winner.append(set_cleaned_up[elements][2])
            click_rate.append(set_cleaned_up[elements][0]/set_cleaned_up[elements][1])
    dict_to_export={
        'headline':headlines,
        'lede':ledes,
        'clicks':clicks,
        'impressions':impressions,
        'winner':winner,
        'click_rate':click_rate
    }
    df=pd.DataFrame(dict_to_export)
    df.to_csv('training_data/cleaned_up_ledes.csv',index=False)
else:
    df=pd.read_csv("./training_data/cleaned_up_ledes.csv")
    headlines=df['headline']
    ledes=df['lede']
    clicks=df['clicks']
    impressions=df['impressions']
    winner=df['winner']
    click_rate=df['click_rate']

#the perplexity score is going to be 1/n summation(i=1 to n) log(P(w_{l+i}|w_{1},w_{2}....w_{l+1},w_{l+2}...,w_{l+i-1}) n is the length of the lede and l is the length of the headline

headlines2=[]
ledes2=[]
clicks2=[]
impressions2=[]
winner2=[]
click_rate2=[]
perplexities=[]
def get_perplexity(values):
    perplexity_score=sum([math.log(value) for value in values])/len(values)
    return -perplexity_score
def get_scores(index):
    sentence=f'{headlines[index]} {ledes[index]}'
    sentence=sentence.split(" ")
    headlines2.append(headlines[index])
    ledes2.append(ledes[index])
    clicks2.append(clicks[index])
    impressions2.append(impressions[index])
    winner2.append(winner[index])
    click_rate2.append(click_rate[index])
    headline_length=headlines[index].count(" ")+1
    probs=[]
    unk=False
    for i in range(headline_length, len(sentence)):
        soi = ' '.join(sentence[0:i])
        input=tokenizer.encode(soi, add_special_tokens=False, return_tensors="pt")
        output=model(input)
        model_output = output[0]
        last_token_prediction = model_output[:, -1]
        last_token_softmax = torch.softmax(last_token_prediction, dim=-1).squeeze()
        index=tokenizer.encode(sentence[i])
        try:
            probs.append(last_token_softmax[index][0])
        except:
            print(index)
            print(sentence)
            print(soi)
            unk=True
            break

    if not unk:
        perplexities.append(get_perplexity(probs))
    else:
        headlines2.pop(-1)
        ledes2.pop(-1)
        impressions2.pop(-1)
        winner2.pop(-1)
        click_rate2.pop(-1)
for i in range(0,len(headlines)):
    if i%50 == 0:
        print(i)
    get_scores(i)
dict_to_export={
        'headline':headlines2,
        'lede':ledes2,
        'perplexity':perplexities,
        'clicks':clicks2,
        'impressions':impressions2,
        'winner':winner2,
        'click_rate':click_rate2
}
df=pd.DataFrame(dict_to_export)
df.to_csv('training_data/perplexities.csv',index=False)
