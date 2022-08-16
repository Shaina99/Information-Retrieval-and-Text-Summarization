import os
import nltk
import time
import random
import string
import pickle
import warnings
import wikipedia
import numpy as np
import pandas as pd
from pprint import pprint
import multiprocessing as mp
from collections import Counter
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from wikipedia import DisambiguationError, PageError
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq  
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from nltk.cluster.util import cosine_distance
import networkx as nx

warnings.filterwarnings('ignore')
random.seed(42)


def download_content(title, ln=100):
    try:
        content = wikipedia.page(title).content
        content = content[:content.find('==')].strip()
    except (DisambiguationError, PageError) as e:
        return None, None
    if len(content.split()) >= ln:
        return title, content
    return None, None

def downloader(k=50, pool_size=10):
    pages_fetch = {}
    complete = False
    with mp.Pool(pool_size) as pool:
        while not complete:
            titles = wikipedia.random(k)
            res = pool.map_async(download_content, titles)
            page = res.get()
            for title, content in page:
                pages_fetch[title] = content
                if len(pages_fetch) > k:
                    complete = True
                    break
    del pages_fetch[None]
    
    return pages_fetch
  
  
t0 = time.time()
n_wiki = 10000
wiki_pickle = 'wiki_content.pkl'
if wiki_pickle not in os.listdir():
    page_dict = downloader(k=n_wiki)
    with open(wiki_pickle, 'wb') as f:
        pickle.dump(page_dict, f)
else:
    with open(wiki_pickle, 'rb') as f:
        page_dict = pickle.load(f)
print(f'{n_wiki} wiki content downloaded in {time.time() - t0}')


def tokenizer(sent):
    tokens = word_tokenize(sent.lower())
    tokens = [w for w in tokens if w not in string.punctuation]
    stemmer = PorterStemmer()
    tokens = list(map(stemmer.stem, tokens))
    lmtzr = WordNetLemmatizer()
    tokens = list(map(lmtzr.lemmatize, tokens))
    return tokens
def pred(vec, X, q):
    y = vec.transform([q])
    res = np.dot(y, X.T).todense()
    res = np.asarray(res).flatten()
    res = np.argsort(res, axis=0)[::-1]
    return res

def evaluate(df, vec, X):
    y_true = []
    y_test = []
    queries = []
    for i, row in df.iterrows():
        random.shuffle(row.top10pct)
        queries.append(' '.join(row.top10pct[:5]))
        y_true.append(i)
        y_test.append(pred(vec, X, queries[-1]))
    group_size = [2, 5, 10]
    recall_k = dict(zip(group_size, [0] * len(group_size)))
    for i in range(len(queries)):
        for gs in group_size:
            recall_k[gs] += 1 if y_true[i] in y_test[i][:gs] else 0
    for gs in group_size:
        recall_k[gs] /= len(df)
    return recall_k

def search_helper(df, vec, X, query, k=5):
    ids = pred(vec, X, query)[:5]
    res = []
    for i in ids:
        res.append((df.iloc[i].title, df.iloc[i].content))
    return res

def main():
    main_df = pd.DataFrame(list(page_dict.items()), columns=['title', 'content'])
    main_df['top10pct'] = None
    for i, row in main_df.iterrows():
        tokens = word_tokenize(row.content.lower())
        stopwords_eng = stopwords.words('english')
        tokens = [w for w in tokens if not (w in stopwords_eng or w in string.punctuation)]
        freq = Counter(tokens)
        top10 = sorted(freq.items(), key=lambda x: -x[1])[:int(len(freq) * 0.3)]
        row.top10pct = [w for w, v in top10]
    
    i = 100

    while i <= len(main_df):
        df = main_df[:i]
        vec = TfidfVectorizer(tokenizer=tokenizer)
        X = vec.fit_transform(df.content)
        recall_k = evaluate(df, vec, X)
        print(f'Recall score for dataset of size {len(df)}.')
        for gs, score in recall_k.items():
            print(f'Recall@{gs}: {score}')
        print()
        i *= 2
        if i > len(main_df) and len(df) < len(main_df):
            i = len(main_df)
            
    min_res = int(input('Input mininum number of results: '))
    query = input('> ')
    ids = pred(vec, X, query)[:min_res]
    for i in ids:
        print(f'{df.iloc[i].title}\n\n{df.iloc[i].content}\n\n\n')
        id=pred(vec, X, query)[:1]
        for i in id:
            text1 = f'{df.iloc[i].title}\n\n{df.iloc[i].content}\n\n\n'
            # Tokenizing the text
            stopWords = set(stopwords.words("english"))
            words = word_tokenize(text1)
   
                # Creating a frequency table to keep the 
               # score of each word
   
            freqTable = dict()
            for word in words:
                word = word.lower()
                if word in stopWords:
                    continue
                if word in freqTable:
                    freqTable[word] += 1
                else:
                    freqTable[word] = 1
                    # Creating a dictionary to keep the score
                    # of each sentence
            sentences = sent_tokenize(text1)
            sentenceValue = dict()
   
            for sentence in sentences:
                for word, freq in freqTable.items():
                    if word in sentence.lower():
                        if sentence in sentenceValue:
                            sentenceValue[sentence] += freq
                        else:
                            sentenceValue[sentence] = freq
   
   
            sumValues = 0
            for sentence in sentenceValue:
                sumValues += sentenceValue[sentence]
                       # Average value of a sentence from the original text
   
            average = int(sumValues / len(sentenceValue))
   
            # Storing sentences into our summary.
            summary = ''
            for sentence in sentences:
                if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
                    summary += " " + sentence
                    
    print(summary)
main()
