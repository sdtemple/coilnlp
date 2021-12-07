# imports #

# nlp libraries
import wordcloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk library
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
enstop = stopwords.words('english')

# gensim library
from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary

# machine learning libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# functions #

# from gensim lda tutorial
def lda(df, num_topics, corpus_type = True, 
        no_below = 2, min_count = 3, min_len = 3, stoppers = enstop,
        chunksize = 1, passes = 50, iterations = 50, eval_every = None,
        alpha = 'symmetric', eta = 'symmetric'):
    
    # typing
    if corpus_type:
        docs = list(df['SENTENCE'])
    else:
        docs = list(df['TEXT'])
        
    # handling NAs    
    docs = [doc for doc in docs if doc is not np.nan]
    # tokenizing
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = tokenizer.tokenize(docs[idx])
    # cleaning                                        
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    docs = [[token for token in doc if len(token) >= min_len] for doc in docs]
    docs = [[token for token in doc if token not in stoppers] for doc in docs]
    # lemmatization                                        
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    # bigrams
    bigram = Phrases(docs, min_count = min_count)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                docs[idx].append(token)
    # filtering, bag of words
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below = no_below, no_above = 1)
    bow = [dictionary.doc2bow(doc) for doc in docs]
    
    # lda parameters
    num_topics = num_topics
    chunksize = chunksize
    passes = passes
    iterations = iterations
    eval_every = eval_every
    temp = dictionary[0] # load dictionary
    id2word = dictionary.id2token
    alpha = alpha
    eta = eta

    model = LdaModel(
        corpus = bow,
        id2word = id2word,
        chunksize = chunksize,
        alpha = alpha,
        eta = eta,
        iterations = iterations,
        num_topics = num_topics,
        passes = passes,
        eval_every = eval_every
    )
    
    return bow, model

def bigram_counts(df, min_count = 3, min_len = 3, stoppers = enstop):
    docs = list(df['TEXT'])
    # tokenizing
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = tokenizer.tokenize(docs[idx])
    # cleaning
    docs = [[token for token in doc if len(token) >= min_len] for doc in docs]
    docs = [[token for token in doc if token not in stoppers] for doc in docs]
    # counting
    bigram = Phrases(docs, min_count = min_count)
    ctr = {}                                        
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                try:
                    ctr[token] += 1
                except KeyError:
                    ctr[token] = 1
    # sorting
    ctr = dict(sorted(ctr.items(), key=lambda item: item[1], reverse = True))
    return ctr
    

def find_word_in_sentences(word, df):
    bools = df['SENTENCE'].apply(lambda x: word in str(x))
    return list(df['SENTENCE'][bools])

def polarity_counts(word, df):
    sentences = find_word_in_sentences(word, df)
    num_neg = 0
    num_neu = 0
    num_pos = 0
    for sentence in sentences:
        pole = TextBlob(sentence).sentiment.polarity
        if pole < 0:
            num_neg += 1
        elif pole == 0:
            num_neu += 1
        else:
            num_pos += 1
    return {'negative':num_neg, 'neutral':num_neu, 'positive':num_pos}

def novel_words(df, wordnet, stopper = enstop):
    words = ' '.join(list(df['TEXT']))
    words = words.split()
    words = set(words)
    novel = []
    for word in words:
        if not wordnet.synsets(word):
            if word not in stopper:
                novel.append(word)
    return novel

def word_counts(df, max_words = 25, ascending = True, stoppers = enstop):
    # word list
    words = ' '.join(list(df['TEXT']))
    words = words.split()
    # word occurrences
    dictionary = {}
    for word in words:
        if word in stoppers:
            pass
        else:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
    # sorting
    words = pd.Series(dictionary)
    words = words.sort_values(ascending = ascending)
    return dict(words[:max_words])

def top_tfidf_terms(text_array, n):
    # term frequency inverse document frequency
    vect = TfidfVectorizer(stop_words = 'english')
    xmat = vect.fit_transform(text_array)
    xmat = xmat.toarray()
    feat = vect.get_feature_names()
    # top n
    nrow = len(text_array)
    for i in range(nrow):
        xrow = xmat[i]
        argp = np.argpartition(xrow, -n)[-n:]
        args = np.argsort(xrow[argp])
        argp = argp[args]
        argp = np.flip(argp)
        dictionary[i] = []
        for j in range(n):
            dictionary[i].append(features[argp[j]])
    return pd.DataFrame.from_dict(dictionary, orient = 'index')

def make_word_cloud(df, max_words = 25, min_len = 3, collocations = True):
    wc = wordcloud.WordCloud(background_color="white", 
                             max_words=max_words, 
                             contour_width=3, 
                             contour_color='steelblue',
                             min_word_length=min_len,
                             collocations=collocations
                            )
    joined_text = ' '.join(list(df['TEXT'].values))
    wc.generate(joined_text)
    return wc

def plot_two_clouds(img, wc1, wc2, title1, title2, loc = 'left', fontsize = 20, pad = 10):
    # first word cloud
    plt.subplot(2, 1, 1)
    plt.imshow(wc1)
    plt.axis('off')
    plt.title(title1, loc = loc, fontsize = fontsize, pad = pad)
    # second word cloud
    plt.subplot(2, 1, 2)
    plt.imshow(wc2)
    plt.axis('off')
    plt.title(title2, loc = loc, fontsize = fontsize, pad = pad)
    # save
    plt.tight_layout()
    plt.savefig(img)
    
def plot_three_clouds(img, wc1, wc2, wc3, title1, title2, title3, loc = 'left', fontsize = 20, pad = 10):
    # first word cloud
    plt.subplot(3, 1, 1)
    plt.imshow(wc1)
    plt.axis('off')
    plt.title(title1, loc = loc, fontsize = fontsize, pad = pad)
    # second word cloud
    plt.subplot(3, 1, 2)
    plt.imshow(wc2)
    plt.axis('off')
    plt.title(title2, loc = loc, fontsize = fontsize, pad = pad)
    # third word cloud
    plt.subplot(3, 1, 3)
    plt.imshow(wc3)
    plt.axis('off')
    plt.title(title3, loc = loc, fontsize = fontsize, pad = pad)
    # save
    plt.tight_layout()
    plt.savefig(img)