#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# In[5]:


# Read in data 

df = pd.read_csv(r"C:\Users\Shant\Downloads\archive (3)\Reviews.csv")
print(df.shape)
df = df.head(500)
print(df.shape)


# In[6]:


df.head()


# In[7]:


#EDA
ax = df['Score'].value_counts().sort_index().plot(kind='bar',title = 'Count of reviews by bars', figsize=(10,5))

ax.set_xlabel('Review Stars')


# In[8]:


## Basic NLTK

example = df['Text'][50]
print(example)


# In[9]:


tokens = nltk.word_tokenize(example)
tokens[:10]


# In[10]:


tagged =nltk.pos_tag(tokens)
tagged[:10]


# In[11]:


entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# In[12]:


#VADER sentiment scoring

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[13]:


sia.polarity_scores('I am so happy')


# In[14]:


sia.polarity_scores('This is the worst thing ever')


# In[15]:


sia.polarity_scores(example)


# In[16]:


# Run the polarity score on the entire data set

res = {}
for i, row in tqdm(df.iterrows(), total = len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)
    


# In[17]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns = {'index': 'Id'})
vaders = vaders.merge(df, how ='left')


# In[18]:


# we have sentiment score and metadata
vaders.head()


# In[19]:


## PLot vader
   
ax = sns.barplot(data = vaders, x = 'Score', y ='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()


# In[20]:


fig, axs = plt.subplots(1,3, figsize =(15,5))

sns.barplot(data = vaders, x = 'Score', y = 'pos',ax= axs[0])
sns.barplot(data = vaders, x = 'Score', y = 'neu',ax = axs[1] )
sns.barplot(data = vaders, x = 'Score', y = 'neg',ax= axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# In[2]:


from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModel
from transformers import AutoModelForSequenceClassification 
from scipy.special import softmax
from transformers import (FlaubertWithLMHeadModel)
from transformers import RobertaConfig, RobertaForSequenceClassification


# In[ ]:


MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[25]:


print(example)
sia.polarity_scores(example)


# In[32]:


#Run for roberta model

encoded_text = tokenizer(example, return_tensors ='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg':scores[0],
    'roberta_neu':scores[1],
    'roberta_pos':scores[2]
}
print(scores_dict)


# In[34]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors ='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg':scores[0],
        'roberta_neu':scores[1],
        'roberta_pos':scores[2]
    }
    return (scores_dict)


# In[46]:


res = {}
for i, row in tqdm(df.iterrows(), total = len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value

        roberta_result = polarity_scores_roberta(text)

        #combining dictionaries vader and roberta results
        both = {**vader_result_rename,**roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')
    


# In[43]:


result = 


# In[47]:


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how = 'left')


# In[50]:


results_df.columns


# In[51]:


# compare scores between models

sns.pairplot(data = results_df, 
             vars= ['vader_neg', 'vader_neu', 'vader_pos',
                   'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue = 'Score', palette = 'tab10')
plt.show()


# In[52]:


results_df.query('Score == 1').sort_values('roberta_pos', ascending = False)['Text'].values[0]


# In[53]:


results_df.query('Score == 1').sort_values('vader_pos', ascending = False)['Text'].values[0]


# In[54]:


# negative 5 star review

results_df.query('Score == 5').sort_values('roberta_neg',ascending = False)['Text'].values[0]


# In[55]:


results_df.query('Score == 5').sort_values('vader_neg', ascending = False)['Text'].values[0]


# In[10]:


from transformers import pipeline

sent_pipeline = pipeline('sentiment-analysis')


# In[9]:


sent_pipeline('I love sentiment analysis')


# In[8]:


sent_pipeline('I want to cry')


# In[ ]:




