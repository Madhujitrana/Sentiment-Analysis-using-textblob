


df=pd.read_csv(r"C:\Users\Madhujit\Desktop\UBER ANALYSIS\flipkart_product.csv",encoding='latin-1')

df

# Take a 100 sample of the dataset for this sentiment analysis

df_100=df[:50]

# Lets clean the text by removing the punctuation,stopword and converting the text to lower.

df_100["summary"]=df_100["Summary"].str.lower()

import re


def function(remove_punctuation):
    pun_text=re.sub('[^\w\s]','',remove_punctuation)
    return pun_text

df_100["clean_text"]=df_100["Summary"].apply(function)

df_100

df_100.drop(["Summary","summary"],axis=1,inplace=True)

df_100

from nltk.corpus import stopwords


stop_word=stopwords.words('english')

def function_1(stop_words):
    fina_data=[i for i in stop_words.split() if i not in stop_word]
    return ' '.join(fina_data)

df_100["cleaned_text"]=df_100["clean_text"].apply(function_1)

df_100

from textblob import TextBlob

def textblob(text):
    text=TextBlob(text).sentiment.polarity
    return text

df_100["polarity_score"]=df_100["cleaned_text"].apply(textblob)

df_100

df_100[df_100["polarity_score"]>0]
    

def assign_label(polarity):
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

df_100["polarity_sub"]=df_100["polarity_score"].apply(assign_label)

df_100

df_100

import plotly.express as py

sentiment=df_100["polarity_sub"].value_counts().reset_index()

fig=py.pie(sentiment,values=sentiment["polarity_sub"],hover_name=sentiment["index"],title="polarity_score")
fig.show()

