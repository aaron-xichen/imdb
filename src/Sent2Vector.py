import sys
sys.path.insert(0, "./libs")
from Word2Vector import Word2Vector
from word2Vec import Word2Vec, Sent2Vec
from sklearn.ensemble import RandomForestClassifier
import copy
import os
import re
import numpy as np
import pandas as pd
import cPickle as pickle

class Sent2Vector:
    def load(self, file_path='save.pickle'):
        file_path = os.path.join("../dataset", file_path)
        data = load_pickle(file_path)
        if data is not None:
            assert len(data) == 7
            self.n_train_labeled = data[0]
            self.n_train_unlabeled = data[1]
            self.n_test_unlabeled = data[2]
            self.test_ids = data[3]
            self.labels = data[4]
            self.reviews_word2vec = data[5]
            self.reviews_sent2vec = data[6]
            return True
        else:
            return False

    def save(self, data, file_path = 'save.pickle'):
        file_path = os.path.join("../dataset", file_path)
        save_pickle(file_path, data)

    def generate_word_sentence_vec(self, word2vec_filename='word.vec', sent2vec_filename='sent.vec'):
        """
        generate word vector based on train_labeled_data, train_unlabeled_data and test_labeled_data
        generate sentence vector based on train_labeled_data and test_labeled_data
        """

        # load pickle if exist
        if self.load():
            return

        word2vec_filename = os.path.abspath(word2vec_filename)
        sent2vec_filename = os.path.abspath(sent2vec_filename)

        # loading all the reviews
        w2v = Word2Vector()
        train_labeled_data = w2v.loadLabeledTrainData()
        train_unlabeled_data = w2v.loadUnlabeledTrainData()
        test_unlabeled_data = w2v.loadTestData()

        self.n_train_labeled = len(train_labeled_data)
        self.n_train_unlabeled = len(train_unlabeled_data)
        self.n_test_unlabeled = len(test_unlabeled_data)

        reviews_labeled_train = train_labeled_data['review']
        reviews_unlabeled_train = train_unlabeled_data['review']
        reviews_unlabeled_test = test_unlabeled_data['review']

        self.test_ids = test_unlabeled_data['id']

        # labels to train classifier
        self.labels = train_labeled_data['sentiment']

        # for generating word vector
        print "...converting labeled training reviews to word sequence"
        self.reviews_word2vec = w2v.convertToWordSequence(reviews_labeled_train)
        # for generating sentence vector
        self.reviews_sent2vec = copy.deepcopy(self.reviews_word2vec)

        print "...converting unlabeled training reviews to word sequence"
        reviews2 = w2v.convertToWordSequence(reviews_unlabeled_train)
        print "...converting unlabeled testing reviews to word sequence"
        reviews3 = w2v.convertToWordSequence(reviews_unlabeled_test)

        self.reviews_word2vec.extend(reviews2)
        self.reviews_word2vec.extend(reviews3)
        self.reviews_sent2vec.extend(reviews3)

        # save pickle
        data = [
            self.n_train_labeled,
            self.n_train_unlabeled,
            self.n_test_unlabeled,
            self.test_ids,
            self.labels,
            self.reviews_word2vec,
            self.reviews_sent2vec
           ]
        self.save(data)

        # generate word2vec
        if not os.path.exists(word2vec_filename):
            print "...generating word vector"
            model = Word2Vec(self.reviews_word2vec, size=300, window=10, sg=0, min_count=5, workers=8)
            model.save(word2vec_filename+ '.model')
            model.save_word2vec_format(word2vec_filename)

        # generate sent2vec
        # this step may be time consuming, please be patient
        if not os.path.exists(sent2vec_filename):
            print "...generateing sentence vector"
            model = Sent2Vec(self.reviews_sent2vec, model_file=word2vec_filename + '.model')
            model.save_sent2vec_format(sent2vec_filename)

    def load_word_sentence_vec(self, word2vec_filename='word.vec', sent2vec_filename='sent.vec'):
        word2vec_filename = os.path.abspath(word2vec_filename)
        sent2vec_filename = os.path.abspath(sent2vec_filename)
        if os.path.exists(word2vec_filename) and os.path.exists(sent2vec_filename):
            self.word_vec = dict()
            # load word2vec
            print "...loading word2vec from {}".format(word2vec_filename)
            with open(word2vec_filename, 'rb') as lines:
                for line_num, line in enumerate(lines):
                    fields = re.split("\\s+", line.strip())
                    # header, skip
                    if len(fields) == 2:
                        line_total_num= int(fields[0])
                        word_vec_len = int(fields[1])
                    else:
                        assert len(fields) == word_vec_len + 1
                        if(line_num % (line_total_num / 10) == 0):
                            print "processing {} of {}".format(line_num, line_total_num)
                        word = fields[0]
                        vec = np.asarray(fields[1:], dtype=np.float32)
                        self.word_vec[word] = vec
                assert line_num == line_total_num

            # load sent2vec
            print "...loading sent2vec from {}".format(sent2vec_filename)
            with open(sent2vec_filename, 'rb') as lines:
                self.sent_vec = []
                for line_num, line in enumerate(lines):
                    fields = re.split("\\s+", line.strip())
                    # header, skip
                    if len(fields) == 2:
                        line_total_num = int(fields[0])
                        sent_vec_len = int(fields[1])
                    else:
                        assert len(fields) == sent_vec_len + 1
                        if(line_num % (line_total_num / 10) == 0):
                            print "processing {} of {}".format(line_num, line_total_num)
                        self.sent_vec.append(fields[1:])
                assert line_num == line_total_num
                self.sent_vec = np.asarray(self.sent_vec, dtype=np.float32)

    # step1: average all word vector in a review
    # step2: concatenate averaged word vector and sentence vector to gain joint features
    # collect info: self.reviews_sent2vec, self.word_vec, self.sent_vec
    def average_concatenate_features(self):
        # check dimension
        assert len(self.reviews_sent2vec) == self.n_train_labeled + self.n_test_unlabeled
        assert len(self.reviews_sent2vec) == self.sent_vec.shape[0]
        assert self.labels.shape[0] == self.n_train_labeled

        # step1
        averaged_features = []
        for review in self.reviews_sent2vec:
            averaged_feature = []
            for word in review:
                if word in self.word_vec:
                    averaged_feature.append(self.word_vec[word])
            averaged_features.append(
                    np.asarray(averaged_feature).mean(axis=0)
                    )
        averaged_features = np.asarray(averaged_features)

        # step2
        assert averaged_features.shape[0] == self.sent_vec.shape[0]
        concatenated_features = np.hstack([averaged_features, self.sent_vec])
        self.features = dict()
        self.features['train'] = concatenated_features[:self.n_train_labeled]
        self.features['test'] = concatenated_features[self.n_train_labeled:]


    # train random forest classifier
    def random_forest_classifier(self):
        forest = RandomForestClassifier(n_estimators=100)
        print "...building RandomForestClassifier"
        forest = forest.fit(self.features['train'], self.labels)

        print "...applying model to testing data"
        result = forest.predict(self.features['test'])
        output = pd.DataFrame( data={"id":self.test_ids, "sentiment":result} )
        saving_path = os.path.join("../prediction", "par_vec.csv")
        saving_path = os.path.expanduser(saving_path)
        saving_path = os.path.abspath(saving_path)
        print "...saving result to " + saving_path
        output.to_csv(saving_path, index=False, quoting=3 )


def load_pickle(file_path, is_verbose = False):
    file_path = os.path.expanduser(file_path)
    file_path= os.path.abspath(file_path)
    if os.path.exists(file_path):
        print "...loading from {}".format(file_path)
        fr = open(file_path, 'rb')
        data = pickle.load(fr)
        fr.close()
        if is_verbose:
            print "summary:{}".format(data)
            print "length:{}".format(len(data))
        return data
    else:
        print 'invalid path:{}'.format(file_path)
        return None

def save_pickle(file_path, data, is_verbose = False):
    file_path = os.path.expanduser(file_path)
    file_path= os.path.abspath(file_path)
    print "...saving to {}".format(file_path)
    if is_verbose:
        print "summary:{}".format(data)
        print "length:{}".format(len(data))
    if not os.path.exists(file_path):
        d = os.path.dirname(file_path)
        if not os.path.exists(d):
            os.makedirs(d)
    fw = open(file_path, 'wb')
    pickle.dump(data, fw, protocol = pickle.HIGHEST_PROTOCOL)
    fw.close()

if __name__ == "__main__":
    s2v = Sent2Vector()
    # generate word vector and sentence vector
    s2v.generate_word_sentence_vec()

    # reload word_vec and sent_vec
    s2v.load_word_sentence_vec()

    s2v.average_concatenate_features()

    s2v.random_forest_classifier()
