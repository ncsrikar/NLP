import pandas as pd
import os
import glob
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
combined = pd.DataFrame(columns=["Reviews"])
files = glob.glob("review texts\\*.csv")
combined = pd.read_csv("combined.csv")
reviews = []
data = pd.DataFrame()
column = []
val = []
pos = 0
neg = 0
neut = 0
pos = {}
neg = {}
for f in files:
    d = pd.read_csv(f)
    for i in d["Review Text"]:
        if(i!="nan"):
            i = str(i)
            text = TextBlob(i)
            k = text.sentiment.polarity
            if(k>0):
                reviews.append(0)
            elif(k<0):
                reviews.append(1)
            elif(k==0):
                reviews.append(0.5)

combined['label'] = reviews 
combined.to_csv("combined.csv",index= False)
combined = pd.read_csv("combined.csv")
combined['tidy_review'] = combined['Review Text'].str.replace("[^a-zA-Z#]", " ")
combined.to_csv("combined.csv",index = False)
dataframe = pd.read_csv("combined.csv")
dataframe['tidy_review'] = dataframe['tidy_review'].astype("str")
for i in dataframe['tidy_review'][dataframe['label']==0.0]:
    i = i.lower()
    blob = TextBlob(i)
    for word,p in blob.tags:
        if(p=="JJ" or p=="JJR" or p=="JJS"):
            if(word not in pos):
                pos[word] = 1
            else:
                pos[word] = pos[word]+1
for i in dataframe['tidy_review'][dataframe['label']==1.0]:
    i = i.lower()
    blob = TextBlob(i)
    for word,n in blob.tags:
        if(n=="JJ" or n=="JJR" or n=="JJS"):
            if(word not in neg):
                neg[word] = 1
            else:
                neg[word] = neg[word]+1
datap = pd.DataFrame({"Words":list(pos.keys()), "Count":list(pos.values())})
datap = datap.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=datap, x= "Words", y = "Count")
ax.set(ylabel = 'Count',title = "Most common Adjectives used in Positive Reviews")
# plt.show()
datan = pd.DataFrame({"Words":list(neg.keys()), "Count":list(neg.values())})
datan = datan.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=datan, x= "Words", y = "Count")
ax.set(ylabel = 'Count',title = "Most common Adjectives used in Negative Reviews")
# plt.show()
wc = WordCloud(background_color = 'white',max_words=10).generate(' '.join(dataframe['tidy_review']))
plt.figure(figsize = (14,8))
plt.imshow(wc)
plt.show()


# print("The Average of all the rating:"+str(avg))
# print("Total Postive Reviews:"+str(pos))
# print("Total Negatibe Reviews:"+str(neg))
# print("Total Neutral Reviews:"+ str(neut))
# plt.hist(val, density=True, bins=10)
# plt.show()


