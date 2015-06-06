__author__ = "Aaron Chen"
__email__  ="aaron.xichen@gmail.com"

import numpy as np
import os
import gzip

import cPickle as pickle
import time
import re


# extract line_num from labels files
def extract_lines(file_name, line_num):
    file_name = os.path.expanduser(file_name)
    file_name = os.path.abspath(file_name)
    if os.path.exists(file_name):
        f = open(file_name, 'rb')
        lines = re.split("\\s+", f.readlines()[0].strip())
        f.close()
        line_num = min(line_num, len(lines))
        lines = lines[:line_num]
        save_file_name = "y_" + str(line_num) + ".txt"
        f = open(save_file_name, 'wb')
        f.writelines(" ".join(lines))
        f.writelines("\n")
        f.close()
    else:
        print "{} not exist".format(file_name)

def load_pickle(file_path, is_verbose = False):
    file_path = os.path.expanduser(file_path)
    file_path= os.path.abspath(file_path)
    if os.path.exists(file_path):
        fr = open(file_path, 'rb')
        data = pickle.load(fr)
        fr.close()
        print "load from {}".format(file_path)
        if is_verbose:
            print "summary:{}".format(data)
            print "length:{}".format(len(data))
        return data
    else:
        print 'invalid path:{}'.format(file_path)

def save_pickle(file_path, data, is_verbose = False):
    file_path = os.path.expanduser(file_path)
    file_path= os.path.abspath(file_path)
    print "saving to {}".format(file_path)
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

def load_feature(path, is_verbose = True, batch_size=500000, interval=50000):
    import theano
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.exists(path):
        print "...reading features from {}".format(path)
        begin = time.clock()
        data = None
        counter = 0
        tmp = []
        with open(path, 'rb') as fr:
            for line in fr:
                counter = counter + 1
                tmp.append(line.strip().split())
                if counter != 0 and counter % batch_size == 0:
                    new_data = np.asarray(tmp, dtype=theano.config.floatX)
                    if data is None:
                        data = new_data
                    else:
                        data = np.vstack([data, new_data])
                    tmp = []
                if counter % (interval) == 0:
                    print "...processed {} in {}s".format(counter, time.clock()-begin)
        if len(tmp) != 0:
            new_data = np.asarray(tmp, dtype=theano.config.floatX)
            if data is None:
                data = new_data
            else:
                data = np.vstack([data, new_data])
        print "finish! elapse: {}s".format(time.clock()-begin)
        return data / 255.
    else:
        print "file {} do not exist".format(path)
        assert 1==2

# read labels
def load_label(path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.exists(path):
        print "...reading labels from {}".format(path)
        f = open(path, 'r')
        lines = f.readline()
        f.close()
        labels = np.asarray(re.split("\\s+", lines.strip()), dtype='int32')
        return labels
    else:
        print "file {} do not exist".format(path)
        assert 1==2

def load_cifar10_soft_label(dir_path = "~/documents/dataset/cifar10", is_verbose=True):
    begin = time.clock()
    dir_path = complement_path(dir_path)
    if check_files_exist(dir_path):
        train_soft_label_path = os.path.join(dir_path, "softmax_train.pickle")
        valid_soft_label_path = os.path.join(dir_path, "softmax_valid.pickle")
        test_soft_label_path = os.path.join(dir_path, "softmax_test.pickle")
        if check_files_exist([train_soft_label_path, valid_soft_label_path, test_soft_label_path]):
            train_set_y = shared_dataset(load_pickle(train_soft_label_path))
            valid_set_y = shared_dataset(load_pickle(valid_soft_label_path))
            test_set_y = shared_dataset(load_pickle(test_soft_label_path))
            elapse = "%0.2f" % (time.clock() - begin)
            if is_verbose:
                print "time elapse:{}s".format(elapse)
            return train_set_y, valid_set_y, test_set_y

def load_cifar10(dir_path = "~/documents/dataset/cifar10", is_wrap = False, is_verbose = True):
    import theano
    dir_path = complement_path(dir_path)
    if os.path.exists(dir_path):
        begin = time.clock()
        # train data
        train_x = None
        train_y = None
        train_valid_data_path = os.path.join(dir_path, 'data_batch_')
        for i in range(1,5):
            dicts = load_pickle(train_valid_data_path + str(i))
            if train_x is None:
                train_x = dicts['data'].astype(theano.config.floatX)
            else:
                train_x = np.vstack([train_x, dicts['data'].astype(theano.config.floatX)])
            if train_y is None:
                train_y = np.array(dicts['labels'], dtype='int32')
            else:
                train_y = np.hstack([train_y, np.asarray(dicts['labels'],dtype='int32')])
        # validate data
        dicts = load_pickle(train_valid_data_path + str(5))
        valid_x = dicts['data'].astype(theano.config.floatX)
        valid_y = np.asarray(dicts['labels'], dtype='int32')
        # test data
        test_data_path = os.path.join(dir_path, "test_batch")
        dicts = load_pickle(test_data_path)
        test_x = dicts['data'].astype(theano.config.floatX)
        test_y = np.asarray(dicts['labels'], dtype='int32')

        if is_wrap:
            train_x, train_y = shared_dataset((train_x, train_y))
            valid_x, valid_y = shared_dataset((valid_x, valid_y))
            test_x, test_y = shared_dataset((test_x, test_y))

        rval = [(train_x, train_y), (valid_x, valid_y),
                (test_x, test_y)]
        elapse = "%0.2f" % (time.clock() - begin)
        if is_verbose:
            print "time elapse:{}s".format(elapse)
        return rval
    else:
        print "file {} does not exitst".format(dir_path)

def load_mnist(file_path="~/documents/dataset/MNIST/mnist.pkl.gz"):
    file_path = complement_path(file_path)
    f = gzip.open(file_path, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    return (train_set, valid_set, test_set)

def load_queries(data_x_path, data_y_path):
    data_x_path = complement_path(data_x_path)
    data_y_path = complement_path(data_y_path)

    data_x = load_feature(data_x_path)
    data_y = load_label(data_y_path)
    test_set_x, test_set_y = shared_dataset((data_x, data_y))
    return test_set_x, test_set_y

# read only normal and abnormal data from given directory
def load_from_dir(data_dir):
    normal_test_x_path = os.path.abspath(os.path.join(data_dir, "normal_test_x.txt"))
    normal_test_y_path = os.path.abspath(os.path.join(data_dir, "normal_test_y.txt"))
    abnormal_test_x_path = os.path.abspath(os.path.join(data_dir, "abnormal_test_x.txt"))
    abnormal_test_y_path = os.path.abspath(os.path.join(data_dir, "abnormal_test_y.txt"))

    if not check_files_exist([normal_test_x_path, normal_test_y_path,
        abnormal_test_x_path, abnormal_test_y_path]):
        return

    print 'loading data from directory {}'.format(data_dir)
    begin = time.clock()

    normal_test_x = load_feature(normal_test_x_path)
    normal_test_y = load_label(normal_test_y_path)
    abnormal_test_x = load_feature(abnormal_test_x_path)
    abnormal_test_y = load_label(abnormal_test_y_path)

    normal_test_x, normal_test_y= shared_dataset((normal_test_x, normal_test_y))
    abnormal_test_x, abnormal_test_y= shared_dataset((abnormal_test_x, abnormal_test_y))

    print "loading elapse:{}s".format(time.clock()-begin)
    return [(normal_test_x, normal_test_y),
            (abnormal_test_x, abnormal_test_y)]

def load_data_raw(data_dir):
    print 'loading data...'
    begin = time.clock()
    if data_dir == 'mnist':
        train_xy, valid_xy, test_xy = load_mnist()
        train_x = train_xy[0]
        train_y = train_xy[1]
        valid_x = valid_xy[0]
        valid_y = valid_xy[1]
        test_x = test_xy[0]
        test_y = test_xy[1]
    else:
        data_dir = os.path.expanduser(data_dir)
        train_x_path = os.path.join(data_dir, "train_x.txt")
        train_y_path = os.path.join(data_dir, "train_y.txt")
        valid_x_path = os.path.join(data_dir, "val_x.txt")
        valid_y_path = os.path.join(data_dir, "val_y.txt")
        test_x_path = os.path.join(data_dir, "test_x.txt")
        test_y_path = os.path.join(data_dir, "test_y.txt")

        train_x = load_feature(train_x_path)
        train_y= load_label(train_y_path)
        print "loading training data done!"
        memory_stat()
        valid_x = load_feature(valid_x_path)
        valid_y = load_label(valid_y_path)
        print "loading validation data done!"
        memory_stat()
        test_x = load_feature(test_x_path)
        test_y = load_label(test_y_path)
        print "loading test data done!"
        memory_stat()

    #shuffle
    train_perm = np.random.permutation(train_x.shape[0])
    valid_perm = np.random.permutation(valid_x.shape[0])
    test_perm = np.random.permutation(test_x.shape[0])
    train_x = train_x[train_perm]
    train_y = train_y[train_perm]
    valid_x = valid_x[valid_perm]
    valid_y = valid_y[valid_perm]
    test_x = test_x[test_perm]
    test_y = test_y[test_perm]


    rval = [(train_x, train_y), (valid_x, valid_y),
            (test_x, test_y)]
    print "loading elapse:{}s".format(time.clock()-begin)
    return rval

# read train_x, train_y, valid_x, valid_y, test_x, test_y
def load_data(data_dir):
    print 'loading data...'
    begin = time.clock()
    if data_dir == 'mnist':
        train_xy, valid_xy, test_xy = load_mnist()
        train_x = train_xy[0]
        train_y = train_xy[1]
        valid_x = valid_xy[0]
        valid_y = valid_xy[1]
        test_x = test_xy[0]
        test_y = test_xy[1]
    else:
        data_dir = os.path.expanduser(data_dir)
        train_x_path = os.path.join(data_dir, "train_x.txt")
        train_y_path = os.path.join(data_dir, "train_y.txt")
        valid_x_path = os.path.join(data_dir, "val_x.txt")
        valid_y_path = os.path.join(data_dir, "val_y.txt")
        test_x_path = os.path.join(data_dir, "test_x.txt")
        test_y_path = os.path.join(data_dir, "test_y.txt")

        train_x = load_feature(train_x_path)
        train_y= load_label(train_y_path)
        valid_x = load_feature(valid_x_path)
        valid_y = load_label(valid_y_path)
        test_x = load_feature(test_x_path)
        test_y = load_label(test_y_path)

    #shuffle
    train_perm = np.random.permutation(train_x.shape[0])
    valid_perm = np.random.permutation(valid_x.shape[0])
    test_perm = np.random.permutation(test_x.shape[0])
    train_x = train_x[train_perm]
    train_y = train_y[train_perm]
    valid_x = valid_x[valid_perm]
    valid_y = valid_y[valid_perm]
    test_x = test_x[test_perm]
    test_y = test_y[test_perm]


    train_set_x, train_set_y = shared_dataset((train_x, train_y))
    valid_set_x, valid_set_y = shared_dataset((valid_x, valid_y))
    test_set_x, test_set_y = shared_dataset((test_x, test_y))

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    print "loading elapse:{}s".format(time.clock()-begin)
    return rval

# borrow from theano
def shared_dataset(data_xy, is_x=True, borrow=True):
    import theano
    import theano.tensor as T
    if isinstance(data_xy, tuple):
        data_x, data_y = data_xy
        shared_x = theano.shared(
                np.asarray(data_x,dtype=theano.config.floatX),
                borrow=borrow)
        shared_y = theano.shared(
                np.asarray(data_y,dtype=theano.config.floatX),
                borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    elif is_x:
        shared_value = theano.shared(
                np.asarray(data_xy,dtype=theano.config.floatX),
                borrow=borrow)
        return shared_value
    else:
        shared_value = theano.shared(
                np.asarray(data_xy,dtype=theano.config.floatX),
                borrow=borrow)
        return T.cast(shared_value, 'int32')

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def complement_path(path = None):
    if path is not None:
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        return path
    else:
        "no path given"

def check_files_exist(file_paths):
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for file_path in file_paths:
        file_path = complement_path(file_path)
        if not os.path.exists(file_path):
            print "file {} do not exist".format(file_path)
            return False
    return True

def memory_stat():
    mem = {}
    f = open("/proc/meminfo")
    lines = f.readlines()
    f.close()
    for line in lines:
        if len(line) < 2: continue
        name = line.split(':')[0]
        var = line.split(':')[1].split()[0]
        mem[name] = long(var) * 1024.0
    mem['MemUsed'] = mem['MemTotal'] - mem['MemFree'] - mem['Buffers'] - mem['Cached']
    mem_total = mem['MemTotal'] / 1024. / 1024. / 1024.
    mem_used = mem['MemUsed'] / 1024. / 1024. / 1024.
    print "MemTotal: %0.2fG, MemUsed: %0.2fG" % (mem_total, mem_used)

def lmdb2images(lmdb_path):
    import matplotlib
    matplotlib.use('Agg')
    import caffe
    import theano
    data = load_from_lmdb(lmdb_path)
    images = []
    for image in data:
        # discard key field, remain value field
        ss = image[1]
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(ss)
        label = datum.label
        # int type data, convert to uint8
        if 0 is not len(datum.data):
            data = np.fromstring(datum.data, dtype=np.uint8)
        # float type data, convert to floatX
        else:
            data = np.asarray(datum.float_data, dtype=theano.config.floatX)
        data.resize(datum.channels, datum.height, datum.width)
        images.append((data, label))
    return images

def imgs2lmdb(imgs, lmdb_path, is_verbose=False):
    import matplotlib
    matplotlib.use('Agg')
    from caffe import io
    values = []
    for img in imgs:
        if img[0].ndim == 2:
            channels = int(1)
        else:
            channels = img[0].shape[2]
        height = img[0].shape[0]
        width= img[0].shape[1]

        img[0].resize(channels, height, width)
        datum = io.array_to_datum(img[0])
        datum.label = int(img[1])
        values.append(datum.SerializeToString())
    if len(values) is not 0:
        dump_to_lmdb(values, lmdb_path, is_verbose, random_order=True)

def load_from_lmdb(file_path):
    import lmdb
    file_path = complement_path(file_path)
    data = []
    if os.path.exists(file_path):
        with lmdb.open(file_path) as env:
            main = env.open_db(dupsort=True)
            with env.begin(db=main) as tnx:
                with tnx.cursor() as cur:
                    for key, value in cur:
                        data.append((key,value))
        return data
    else:
        print "lmdb file {} not exist".format(file_path)


def dump_to_lmdb(values, file_path, is_verbose= False, random_order=True):
    import lmdb
    file_path = complement_path(file_path)
    if not isinstance(values, list):
        print "{} is not a list, do nothing".format(values)
        return
    if os.path.exists(file_path):
        if is_verbose:
            print "detects {}, use it".format(file_path)
    else:
        print "{} is not exist, create new one".format(file_path)
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    with lmdb.open(file_path, map_size=1099511627776)  as env:
        with env.begin(write=True) as txn:
            with txn.cursor() as cur:
                if random_order:
                    for v in values:
                        is_ok = False
                        while not is_ok:
                            try:
                                key = '{:010}'.format(np.random.randint(0, 2**32))
                                is_ok = cur.put(key, v, overwrite=False)
                            except Exception, e:
                                print type(key)
                                print type(v)
                                print v
                                raise e
                else:
                    for i,v in enumerate(values):
                        key = '{:010}'.format(i)
                        assert cur.put(key, v, overwrite=False)

        print "#new records:{}".format(len(values))
        print "#total records:{}".format(env.stat()['entries'])
    if is_verbose:
        print "put data {} sucessfully".format(values)

def show_status(lmdb_path):
    import lmdb
    lmdb_path = complement_path(lmdb_path)
    if os.path.exists(lmdb_path):
        with lmdb.open(lmdb_path, map_size=1099511627776) as env:
            print env.stat()['entries']
    else:
        print "lmdb_file {} is not exist, do nothing".format(lmdb_path)

def pickle2lmdb(pickle_path, lmdb_path):
    data = load_pickle(pickle_path)
    np2lmdb(data, lmdb_path)

def np2lmdb(data, lmdb_path, labels=None, shape=None):
    import matplotlib
    matplotlib.use('Agg')
    from caffe import io
    records = []
    assert isinstance(data, np.ndarray)
    if labels is not None:
        assert len(data) == len(labels)
        assert isinstance(labels[0], np.int)

    for i, each in enumerate(data):
        try:
            if shape is None:
                each.resize(each.shape[0], 1, 1)
            else:
                each.resize(shape)
            one = io.array_to_datum(each)
            if labels is not None:
                one.label = labels[i]
            records.append(one.SerializeToString())
        except Exception, e:
            print each.shape
            raise e
    dump_to_lmdb(records, lmdb_path, random_order=False)















