#!/usr/bin/env python
# encoding: utf-8

import Word2Vector
import utils
import enchant

d = enchant.Dict("en_US")
w2v = Word2Vector.Word2Vector()
train = w2v.loadLabeledTrainData()
test = w2v.loadTestData()
train_reviews = train['review']
train_labels = train['sentiment']
test_reviews = test['review']
clean_train_reviews = w2v.convertToWordSequence(train_reviews)
clean_test_reviews = w2v.convertToWordSequence(test_reviews)
dicts = {}
for review in clean_train_reviews:
    for word in review:
        if d.check(word) and not dicts.has_key(word):
            dicts[word] = len(dicts)
for review in clean_test_reviews:
    for word in review:
        if d.check(word) and not dicts.has_key(word):
            dicts[word] = len(dicts)


encode_train_reviews = []
for review in clean_train_reviews:
    each = []
    for word in review:
        if dicts.has_key(word):
            each.append(dicts[word])
    if len(each) != 0:
        encode_train_reviews.append(each)

encode_test_reviews = []
for review in clean_test_reviews:
    each = []
    for word in review:
        if dicts.has_key(word):
            each.append(dicts[word])
    if len(each) != 0:
        encode_test_reviews.append(each)

assert len(encode_train_reviews) == len(train_labels)
train = (encode_train_reviews, train_labels)

utils.save_pickle("encode_train_reviews.pickle", train)
utils.save_pickle("encode_test_reviews.pickle", encode_test_reviews)
utils.save_pickle("dicts.pickle", dicts)
