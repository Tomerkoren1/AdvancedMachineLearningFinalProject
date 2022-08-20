import os

import gensim
import numpy as np

from gensim.models import CoherenceModel
import nltk
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
# from src.topic_model.topic_loader import get_topic_model_folder, get_topic_parent_folder



class LDA():
    def __init__(self, num_topics = 80, init_data_words = [''], to_evaluate= False):

        self.num_topics = num_topics
        print("Init data words preprocessing")
        corpus, self.id2word, processed_texts = self.lda_preprocess(init_data_words)
        self.model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                              id2word=self.id2word,
                                              num_topics=num_topics,
                                              random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

        if to_evaluate:
            self.evaluate_lda_topic_model()
    
    def remove_stopwords(self, texts):
        stop_words = self.stopwordsList()
        return [[word for word in simple_preprocess(str(doc)) 
                if word not in stop_words] for doc in texts]
                
    def lda_preprocess(self, data_tokenized, id2word=None, delete_stopwords=True):
        '''
        Preprocess tokenized text data for LDA (deleting stopwords, recognising ngrams, lemmatisation
        :param data_tokenized: tokenized text data as nested list of tokens
        :param id2word: preexisting id2word object (to map dev or test split to identical ids), otherwise None (train split)
        :param print_steps: print intermediate output examples
        :return: preprocessed corpus as bag of wordids, id2word
        '''
        # assert type(data_tokenized)==list
        # assert type(data_tokenized[0])==list
        # assert type(data_tokenized[0][0])==str
        # data_finished = [self.removeShortLongWords(s) for s in data_tokenized]
        # if delete_stopwords:
        #     # print('removing stopwords')
        #     data_finished = [self.removeStopwords(s) for s in data_finished]
        data_finished = self.remove_stopwords(data_tokenized)

        if id2word is None:
            # Create Dictionary
            id2word = corpora.Dictionary(data_finished)

        # Create Corpus
        processed_texts = data_finished

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in processed_texts]

        return corpus, id2word, processed_texts
    
    @staticmethod
    def stopwordsList():
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.append('...')
        stopwords.append('___')
        stopwords.append('<url>')
        stopwords.append('<img>')
        stopwords.append('<URL>')
        stopwords.append('<IMG>')
        stopwords.append("can't")
        stopwords.append("i've")
        stopwords.append("i'll")
        stopwords.append("i'm")
        stopwords.append("that's")
        stopwords.append("n't")
        stopwords.append('rrb')
        stopwords.append('lrb')
        stopwords.extend(['from', 'subject', 're', 'edu', 'use'])
        return stopwords
    
    def infer_lda_topics(self, new_corpus):
        '''
            Function to infer topic distribution
        '''
        assert type(new_corpus) == list # of sentences
        assert type(new_corpus[0]) == list # of word id/count tuples
        assert type(new_corpus[0][0]) == tuple
        return self.model[new_corpus]


    def infer_topic_dist(self, new_corpus):
        '''
        Get global topic distribution of all sentences from dev or test set from lda_model
        '''

        # print('Inferring topic distribution...')
        # obtain topic distribution for dev set
        assert type(new_corpus) == list # of sentences
        assert type(new_corpus[0]) == list # of word id/count tuples
        assert type(new_corpus[0][0]) == tuple

        dist_over_topic = self.infer_lda_topics(new_corpus)
        # extract topic vectors from prediction
        global_topics = self.extract_topic_from_lda_prediction(dist_over_topic)
        # sanity check
        assert (len(global_topics.shape) == 2)
        assert (global_topics.shape[1] == self.num_topics)
        # print(global_topics.shape)
        # print('Done.')
        return global_topics

# -----------------------------------------------
# functions to explore topic distribution
# -----------------------------------------------

    def extract_topic_from_lda_prediction(self, dist_over_topic):
        '''
        Extracts topic vectors from prediction
        :param dist_over_topic: nested topic distribution object
        :param num_topics: number of topics
        :return: topic array with (examples, num_topics)
        '''
        # iterate through nested representation and extract topic distribution as single vector with length=num_topics for each document
        topic_array = []
        for j, example in enumerate(dist_over_topic):
            i = 0
            topics_per_doc = []
            for topic_num, prop_topic in example[0]:
                # print(i)
                # print(topic_num)
                while not i == topic_num:
                    topics_per_doc.append(0)  # fill in 'missing' topics with probabilites < threshold as P=0
                    # print('missing')
                    i = i + 1
                topics_per_doc.append(prop_topic)
                i = i + 1
            while len(topics_per_doc) < self.num_topics:
                topics_per_doc.append(0)  # fill in last 'missing' topics
            topic_array.append(np.array(topics_per_doc))
        global_topics = np.array(topic_array)
        # sanity check
        if not ((len(global_topics.shape) == 2) and (global_topics.shape[1] == self.num_topics)):
            print('Inconsistent topic vector length detected:')
            i = 0
            for dist, example in zip(global_topics, dist_over_topic):
                if len(dist) != self.num_topics:
                    print('{}th example with length {}: {}'.format(i, len(dist), dist))
                    print('from: {}'.format(example[0]))
                    print('--')
                i += 1
        return global_topics


    def evaluate_lda_topic_model(self, corpus, processed_texts, id2word):
        '''
        Performs intrinsic evaluation by computing perplexity and coherence
        :param lda_model:
        :param corpus:
        :param processed_texts:
        :param id2word:
        :return: dictionary with perplexity and coherence
        '''
        try:
            model_perplexity = self.model.log_perplexity(corpus)
            print('\nPerplexity: ', model_perplexity)  # a measure of how good the model is. lower the better.
        except AttributeError:
            pass
        coherence_model_lda = CoherenceModel(model=self.model, texts=processed_texts, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()  # the higher the better
        print('\nCoherence Score: ', coherence_lda)



