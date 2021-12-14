# imports #

# nlp libraries
import wordcloud
from textblob import TextBlob

# sklearn library
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

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

# modified from gensim lda tutorial
def lda(df, 
        num_topics, 
        corpus_type = True, 
        no_below = 2, 
        min_count = 3, 
        min_len = 3, 
        stoppers = enstop,
        chunksize = 1000, 
        passes = 100, 
        iterations = 500, 
        eval_every = 1,
        alpha = 'auto', 
        eta = 'auto'):
    '''
    Fit a latent Dirichlet allocation model
    
    :param df: documents or sentences corpus
    :type df: pandas DataFrame
    :type num_topics: int
    :param corpus_type: documents if True else sentences
    :param no_below: term in at least so many documents
    :type no_below: int
    :param min_count: filter for count occurrences
    :type min_count: int
    :param min_len: filter for word length
    :type min_len: int
    :param stoppers: stopwords
    
    ... see gensim.models.LdaModel ...
    
    :returns: bag-of-words, topic model
    :rtype: tuple (doc2bow object, LdaModel object)
    '''
    
    # typing
    if corpus_type:
        docs = list(df['TEXT'])
    else:
        docs = list(df['SENTENCE'])
        
        
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
    
    # n-grams
    grams = Phrases(docs, min_count = min_count)
    for idx in range(len(docs)):
        for token in grams[docs[idx]]:
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

def lda_topics(mdl, bow, topn = 10):
    '''
    Summarize the LDA topics
    
    :param mdl: fit LDA model
    :type mdl: LdaModel object
    :param bow: bag-of-words
    :type bow: doc2bow object
    :param topn: words from topic
    :type topn: int
    
    :returns: topic summary table
    :rtype: pandas DataFrame
    '''
    
    output = mdl.top_topics(bow, topn = topn)
    output = {i:[x[1] for x in output[i][0]] for i in range(len(output))}
    output = pd.DataFrame(output)
    output.columns = ['Topic ' + str(i+1) for i in range(output.shape[1])]
    
    return output

# modified from Medium blog "Topic Modeling with LDA and NMF on the ABC News Headlines dataset"
# https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df
def nmf(df, 
        num_topics, 
        corpus_type = True, 
        min_count = 3, 
        min_len = 3, 
        stoppers = enstop,
        max_features = 10_000,
        smooth_idf = False,
        norm = 'l1',
        init = 'nndsvd'
       ):
    '''
    Fit a non-negative matrix factorization model
    
    :param df: documents or sentences corpus
    :type df: pandas DataFrame
    :type num_topics: int
    :param corpus_type: documents if True else sentences
    :param min_count: filter for count occurrences
    :type min_count: int
    :param min_len: filter for word length
    :type min_len: int
    :param stoppers: stopwords
    
    ... see CountVectorizer, TfidfTransformer, normalize, nmf in sklearn ...
    
    :returns: topic model
    :rtype: tuple (CountVectorizer object, NMF object)
    '''
    
    # typing
    if corpus_type:
        docs = list(df['TEXT'])
    else:
        docs = list(df['SENTENCE'])
        
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
    docs = [' '.join(doc) for doc in docs]
    
    # vectorizing
    vectorizer = CountVectorizer(analyzer = 'word', max_features = max_features)
    x_counts = vectorizer.fit_transform(docs)
    
    # tfidf
    transformer = TfidfTransformer(smooth_idf = smooth_idf)
    x_tfidf = transformer.fit_transform(x_counts)
    
    # normalize
    x_tfidf_norm = normalize(x_tfidf, norm = norm, axis = 1)
    
    # modeling
    mdl = NMF(n_components=num_topics, init = init)
    mdl.fit(x_tfidf_norm)
    
    return vectorizer, mdl

def nmf_topics(mdl, vectorizer, num_topics, topn):
    '''
    Summarize the NMF topics
    
    :param mdl: non-negative matrix factorization
    :type mdl: NMF object
    :param vectorizer: count vectorizer
    :type vectorizer: CountVectorizer
    :type num_topics: int
    :param topn: words from topic
    :type topn: int
    
    :returns: topic summary table
    :rtype: pandas DataFrame
    '''
    
    feat_names = vectorizer.get_feature_names()
    word_dict = {};
    for i in range(num_topics):
        words_ids = mdl.components_[i].argsort()[:-topn - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic ' + str(i+1)] = words

    return pd.DataFrame(word_dict);

def ngrams(df, min_count = 3, min_len = 3, stoppers = enstop):
    '''
    Find n-grams in corpus
    
    :param df: sentences corpus
    :type df: pandas DataFrame
    :param min_count: filter for count occurrences
    :type min_count: int
    :param min_len: filter for word length
    :type min_len: int
    :param stoppers: stopwords
    
    :returns: sorted ngram occurrences
    :rtype: dict
    '''
    
    # nan filtering
    texts = df['SENTENCE']
    bools = []
    for i in range(len(texts)):
        try:
            np.isnan(texts[i])
        except TypeError:
            bools.append(i)
    texts = texts[bools]
    texts = list(texts)
    
    # tokenizing
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(texts)):
        try:
            np.isnan(texts[idx])
        except TypeError:
            texts[idx] = tokenizer.tokenize(texts[idx])      
            
    # text data cleaning
    texts = [[token for token in text if len(token) >= min_len] for text in texts]
    texts = [[token for token in text if token not in stoppers] for text in texts]
    
    # counting
    ngrams = Phrases(texts, min_count = min_count)
    output = {}                                        
    for idx in range(len(texts)):
        for token in ngrams[texts[idx]]:
            if '_' in token:
                try:
                    output[token] += 1
                except KeyError:
                    output[token] = 1
                    
    # sorting
    output = dict(sorted(output.items(), key=lambda item: item[1], reverse = True))
    
    return output
    

def find_word_in_sentences(word, df, n_context = 1):
    '''
    Find sentences from a corpus that contain input word
    
    :param word: input word
    :type word: str
    :param df: sentences corpus
    :type df: pandas DataFrame
    :param n_context: contextual lines before and after
    :type n_context: int
    
    :returns: Sorted list of annotated texts
    :rtype: list of tuples (topic, state, id, polarity, subjectivity, text)
    '''
    
    # setup
    nrow = df.shape[0]
    bools = df['SENTENCE'].apply(lambda x: word in str(x))
    indices = [i for i, x in enumerate(bools) if x]
    
    # loop
    output = []
    for i in indices:
        row = df.iloc[i]
        topic = row['TOPIC']
        state = row['STATE']
        docid = row['DOCNUM']
        lines = row['SENTENCE']
        
        # context before
        for j in range(i - 1, i - n_context - 1, -1):
            if (j >= 0):
                row = df.iloc[j]
                if (row['TOPIC'] == topic) and (row['STATE'] == state) and (row['DOCNUM'] == docid):
                    try:
                        np.isnan(row['SENTENCE'])
                    except TypeError:
                        lines = row['SENTENCE'] + ' . ' + lines
                    
        # context after
        for j in range(i + 1, i + n_context + 1, 1):
            if j <= (nrow - 1):
                row = df.iloc[j]
                if (row['TOPIC'] == topic) and (row['STATE'] == state) and (row['DOCNUM'] == docid):
                    try:
                        np.isnan(row['SENTENCE'])
                    except TypeError:
                        lines = lines + ' . ' + row['SENTENCE']
        
        # sentiment analysis
        tb = TextBlob(lines)
        pol = tb.sentiment.polarity
        sub = tb.sentiment.subjectivity
        
        # append to output                
        output.append((topic, state, docid, pol, sub, lines))
        
        # sort output
        output.sort(key = lambda x : x[3])
        
    return output

def polarity_counts(word, df):
    '''
    Count TextBlob polarity for sentences containing word
    
    :type word: str
    :param df: sentences corpus
    :type df: pandas DataFrame
    
    :returns: counts
    :rtype: dict
    '''
    
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

def novel_words(df, wordnet = wordnet, stopper = enstop):
    '''
    Find novel words not in a dictionary
    
    :param df: documents corpus
    :type df: pandas DataFrame
    :param wordnet: some dictionary
    :param stopper: stopwords
    
    :returns: novel words
    :rtype: list
    '''
    
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
    '''
    Compute word counts
    
    :param df: documents corpus
    :type df: pandas DataFrame
    :param max_words: words in cloud
    :type max_words: int
    :param ascending: sorting option
    :type ascending: bool
    :param stoppers: stopwords
    
    :returns: word counts
    :rtype: dict
    '''
    
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

def top_tfidf_terms(df, n):
    '''
    Compute top term-frequency inverse-document-frequency items per document
    
    :param df: documents corpus
    :type df: pandas DataFrame
    :param n: items per row
    :type n: int
    
    :returns: terms matrix
    :rtype: pandas DataFrame
    '''
    
    # term frequency inverse document frequency
    text_array = df['TEXT']
    vect = TfidfVectorizer(stop_words = 'english')
    xmat = vect.fit_transform(text_array)
    xmat = xmat.toarray()
    feat = vect.get_feature_names()
    
    # top n
    nrow = len(text_array)
    dictionary = {}
    for i in range(nrow):
        xrow = xmat[i]
        argp = np.argpartition(xrow, -n)[-n:]
        args = np.argsort(xrow[argp])
        argp = argp[args]
        argp = np.flip(argp)
        dictionary[i] = list(df.iloc[i][:3])
        for j in range(n):
            dictionary[i].append(feat[argp[j]])
            
    return pd.DataFrame.from_dict(dictionary, 
                                  orient = 'index',
                                  columns = ['Topic', 'State', 'DocID'] + ['Tfidf ' + str(i+1) for i in range(n)]
                                 )

def make_word_cloud(df, max_words = 25, min_len = 3, collocations = True):
    '''
    Make word cloud
    
    :param df: documents corpus
    :type df: pandas DataFrame
    :param max_words: words in cloud
    :type max_words: int
    :param min_len: filter for word length
    :type min_len: int
    :param collocations: consider ngrams
    :type collocations: bool
    
    :returns: word cloud
    :rtype: WordCloud object
    '''
    
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

def ngram_cloud(grams, max_words = 15):
    '''
    Make ngram cloud
    
    :param grams: ngram occurrences
    :type grams: dict
    :param max_words: words in cloud
    :type max_words: int
    
    :returns: word cloud
    :rtype: WordCloud object
    '''
    
    grams = [(x + ' ') * i for x, i in grams.items()]
    grams = ' '.join(grams)
    wc = wordcloud.WordCloud(background_color="white", 
                             max_words=max_words, 
                             contour_width=3, 
                             contour_color='steelblue',
                             collocations = False
                            )
    wc.generate(grams)
    
    return wc

def plot_two_clouds(img, wc1, wc2, title1, title2, loc = 'left', fontsize = 20, pad = 10):
    '''
    Juxtapose two WordCloud diagrams
    
    :param img: file name
    :type img: str
    :param wc1, wc2: word clouds
    :type wc1, wc2: WordCloud objects
    :param title1, title2: titles
    :type title1, title2: str
    '''
    
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
    '''
    Juxtapose three WordCloud diagrams
    
    :param img: file name
    :type img: str
    :param wc1, wc2, wc3: word clouds
    :type wc1, wc2, wc3: WordCloud objects
    :param title1, title2, title3: titles
    :type title1, title2, title3: str
    '''
    
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