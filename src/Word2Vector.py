import sys
sys.path.insert(0, "./libs")
import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import time

class Word2Vector:
    def __init__(self):
        self.labeled_train_data_path = "../dataset/labeledTrainData.tsv"
        self.test_data_path = "../dataset/testData.tsv"
        self.unlabeled_train_data_path = "../dataset/unlabeledTrainData.tsv"
        self.sample_submission_path = "../dataset/sampleSubmission.csv"
        self.model_save_dir= "../model"
        self.prediction_dir="../prediction"
        self.dataset_dir = "../dataset"

    # load data
    def loadLabeledTrainData(self):
        print "...loading labeled train data"
        print "...finished"
        return pd.read_csv(self.labeled_train_data_path, header=0, delimiter="\t", quoting=3, encoding='utf-8')
    def loadUnlabeledTrainData(self):
        print "...loading unlabeled train data"
        print "...finished"
        return pd.read_csv(self.unlabeled_train_data_path, header=0, delimiter="\t", quoting=3, encoding='utf-8')
    def loadTestData(self):
        print "...loading test data"
        print "...finished"
        return pd.read_csv(self.test_data_path, header=0, delimiter="\t", quoting=3, encoding='utf-8')
    def loadSampleSubmission(self):
        print "loading sample submission data"
        return pd.read_csv(self.sample_submission_path, header=0, delimiter="\t", quoting=3, encoding='utf-8')

    # split a sentence to word list
    def sentenceToWordList(self, review, remove_stopwords=False):
        review_text = BeautifulSoup(review).getText()
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        return(words)


    # convert a review to separate sentences, each sentence is consisted of word sequences
    def reviewToSentences(self, review, tokenizer, remove_stopwords=False):
        raw_sentences = tokenizer.tokenize(review.strip())
        sentences= []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                s = self.sentenceToWordList(raw_sentence, remove_stopwords)
                sentences.append(s)
        return sentences

    # convert all the reviews to seprate sentences, each sentence is consited of word sequences
    def extractSentences(self, sentences_list, \
            tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')):
        print "begin to extract sentences"
        size = len(sentences_list)
        i = 0
        sentences = []
        for sentence in sentences_list:
            i = i + 1
            sentences += self.reviewToSentences(sentence, tokenizer)
            if i%100 == 0:
                print "process {} of {}".format(i, size)
        return sentences

    # construct model
    def modelTrain(self, sentences, num_workers=4, num_features=300, min_word_count=5,
            context=10, downsampling=1e-3):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        #begin training model
        print "Training..."
        model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count=min_word_count, \
                window=context, sample=downsampling)
        model.init_sims(replace=True)
        model_name = "{}features_{}minwords_{}context".format(num_features, min_word_count, context)
        model.save(model_name)
        return model

    # average all the words vectors in the given word list by their vectors
    # generate the vector representation of a review
    def makeFeatureVec(self, words, model, num_features):
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0
        index2word_set = set(model.index2word)

        for word in words:
            if word in index2word_set:
                nwords = nwords + 1
                featureVec = np.add(featureVec, model[word])

        featureVec = np.divide(featureVec, nwords)
        return featureVec

    # generate the vector representation of all the reviews
    def getAvgFeatureVecs(self, reviews, model, num_features):
        counter = 0
        reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
        for review in reviews:
            if counter%1000. == 0:
                print "Review {} of {}".format(counter, len(reviews))
            reviewFeatureVecs[counter] = self.makeFeatureVec(review, model, num_features)
            counter = counter + 1
        return reviewFeatureVecs

    # first generate the centroids code book
    # then return the ids of the centroids of corresponding vocabulary
    def getWordCentroidMap(self, model, num_clusters):
        start = time.time()

        word_vectors = model.syn0
        print "begin to clustering to gaining code book"
        kmeans_clustering = KMeans(n_clusters = num_clusters)
        idx = kmeans_clustering.fit_transform(word_vectors)

        end = time.time()
        elapsed= end - start
        print "Time takes for K means clustering: {} seconds".format(elapsed)
        return dict(zip(model.index2word, idx))

    # return the apperance of each centroids in the code book, forming a vecotor representation
    # the length of the vector equals the size of the centroids code boo
    def createBagOfCentriods(self, wordlist, word_centroid_map):
        num_centroids = max(word_centroid_map.values()) + 1
        bag_of_centroids = np.zeros(num_centroids, dtype='float32')
        for word in wordlist:
            if word in word_centroid_map:
                index = word_centroid_map[word]
                bag_of_centroids[index] += 1
        return bag_of_centroids

    # is already exist model?
    def isModelExist(self):
        pattern = "\d+features_\d+minwords_\d+context$"
        flist = os.listdir(self.model_save_dir)
        for f in flist:
            if re.match(pattern, f):
                return True, f
        return None, None


    # generate word vector, return the model
    def generateWordVectorModel(self, train_review, test_review, isRecalculate=False):
        print "generating model..."
        is_model_exist, model_path = self.isModelExist()
        if not isRecalculate and is_model_exist:
            print "finding existing model, return directily"
            return word2vec.Word2Vec.load(self.model_save_dir + os.path.sep + model_path)

        # construct the model
        print "do not find exist model, begin to build model"
        sentences = []
        sentences += self.extractSentences(train_review)
        sentences += self.extractSentences(test_review)
        return self.modelTrain(sentences)

    # convert each review to word sequence
    def convertToWordSequence(self, reviews, remove_stopwords=True):
        clean_reviews = []
        counter = 0
        for review in reviews:
            if counter % (len(reviews) // 10)== 0:
                print "processed {} of {}".format(counter, len(reviews))
            clean_reviews.append(self.sentenceToWordList(review, remove_stopwords))
            counter += 1
        return clean_reviews


    # vector Averaging Approach
    def vectorAveragingApproach(self, clean_train_reviews, clean_test_reviews, model, num_features):
        print "running vector averaging approach..."
        trainDataVecs = self.getAvgFeatureVecs(clean_train_reviews, model, num_features)
        testDataVecs= self.getAvgFeatureVecs(clean_test_reviews, model, num_features)

        forest = RandomForestClassifier(n_estimators = 100)
        print "building RandomForestClassifier "
        forest = forest.fit(trainDataVecs, train["sentiment"])

        print "apply model to testing data"
        result = forest.predict(testDataVecs)

        output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
        saving_path = self.prediction_dir + os.path.sep + "Word2Vec_AverageVectors.csv"
        print "saving result to " + saving_path
        output.to_csv(saving_path, index=False, quoting=3 )

    # Centroids Clustering Approach
    def centroidsClustringApproach(self, clean_train_reviews, clean_test_reviews, model):
        print "running centroids clustering approach"
        word_vectors = model.syn0
        num_clusters = word_vectors.shape[0]/5
        word_centroid_map = self.getWordCentroidMap(model, num_clusters)
        train_centriods = np.zeros((train["review"].size, num_clusters), dtype='float32')
        test_centriods = np.zeros((test["review"].size, num_clusters), dtype='float32')
        counter = 0
        print "creating bag of centriods"
        for review in clean_train_reviews:
            if counter%100 == 0:
                print "process {} of {}".format(counter, len(clean_train_reviews))
            train_centriods[counter] = self.createBagOfCentriods(review, word_centroid_map)
            counter += 1
        counter = 0
        for review in clean_test_reviews:
            if counter%100 == 0:
                print "process {} of {}".format(counter, len(clean_test_reviews))
            test_centriods[counter] = self.createBagOfCentriods(review, word_centroid_map)
            counter += 1

        forest = RandomForestClassifier(n_estimators=100)
        print "building RandomForestClassifier for clustering approach"
        forest = forest.fit(train_centriods, train["sentiment"])
        print "applying model to tesing data"
        result = forest.predict(test_centriods)
        output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
        saving_path = self.prediction_dir + os.path.sep + "BagOfCentriods.csv"
        print "saving result to " + saving_path
        output.to_csv(saving_path, index=False, quoting=3 )

if __name__ == "__main__":
    w2v = Word2Vector()
    # load original data
    train = w2v.loadLabeledTrainData()
    test = w2v.loadTestData()

    # generate word vector model
    model = w2v.generateWordVectorModel(train["review"], test["review"])

    # feature num of each word in the vocabulary
    num_features = model.syn0.shape[1]

    # appraoch controller
    vector_averaging_approach = True
    centroids_clustring_approach = True

    # clean data
    print "convering training reviews to word sequences"
    clean_train_reviews = w2v.convertToWordSequence(train["review"])
    print "converting testing reviews to word sequences"
    clean_test_reviews = w2v.convertToWordSequence(test["review"])

    # Vector Averaging Approach
    if vector_averaging_approach:
        w2v.vectorAveragingApproach(clean_train_reviews, clean_test_reviews, model, num_features)

    # Centroids Clustering Approach
    if centroids_clustring_approach:
        w2v.centroidsClustringApproach(clean_train_reviews, clean_test_reviews, model)
