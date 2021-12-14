# coilnlp

This package contains Python functions for text data analysis. These functions are mostly sorting, list comprehensions, and methods from NLP libraries bundled into a user-friendly and documented form. Their purpose is to assist teachers and linguists that I consulted for in the analysis of blog posts. Main function is `find_word_in_sentences()`. Based on other functions, linguists and teachers are expected to search for specific words in the sentences corpus. Tidying up the output can be achieved via list comprehensions. (Functions expect specific dataframes.) In sum, this package promotes search informed by word frequencies and topic modeling, thereby leaving interpretation to the linguists and teachers. 

Example Analyses:
* `not_informed_analysis.ipynb` : generic text data analysis
* `outcome_informed_analysis.ipynb` : text data analysis informed by known learning outcomes
* words

Console:
* `build_docs()` : build documents corpus
* `build_sent()` : build sentences corpus

Functions:
* `find_word_in_sentences()` : word search in corpus
* `word_counts()` : word counts
* `ngrams()` : compound words; word associations
* `make_word_cloud()` : word cloud
* `top_tfidf_terms()` : most/least frequent words per document
* `novel_words()` : words not in wordnet
* `lda()` : latent Dirichlet allocation
* `lda_topics()` : topics from LDA model
* `nmf()` : non-negative matrix factorization
* `nmf_topics()` : topics from NMF model
* `polarity_counts()` : polarity from TextBlob rule-based sentiment

The consulting project was to analyze text data from an online discussion board. Participants to the discussion board came from multicultural and multilingual backgrounds. They all participated in a unique online learning experience:

COIL (Collaborative Online International Learning) is an approach to fostering global competence through development of a multicultural learning environment that links university classes in different countries. Using various communication technologies, students complete shared assignments and projects, with faculty members from each country co-teaching and managing coursework. Students attended universities in the USA and in Iraq.
