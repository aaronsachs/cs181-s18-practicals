## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test directories and parse each xml file into an xml.etree.ElementTree, 
## which is a standard python object used to represent an xml file in memory.
## (More information about xml.etree.ElementTree objects can be found here:
## http://docs.python.org/2/library/xml.etree.elementtree.html
## and here: http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each ElementTree object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code that randomly picks class-specific weights and predicts
## the class with the weights that give the highest score. If your prediction
## algorithm involves class-specific weights, you should, of course, learn 
## these class-specific weights in a more intelligent way.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take an ElementTree object representing
## an xml file (which contains, among other things, the sequence of system calls a
## piece of potential malware has made), and returns a dictionary mapping feature names to 
## their respective numeric values. 
## For instance, a simple feature-function might map a system call history to the
## dictionary {'first_call-load_image': 1}. This is a boolean feature indicating
## whether the first system call made by the executable was 'load_image'. 
## Real-valued or count-based features can of course also be defined in this way. 
## Because this feature-function will be run over ElementTree objects for each 
## software execution history instance, we will have the (different)
## feature values of this feature for each history, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions (for a particular
## training example) will be unioned, so we can collect all the feature values 
## associated with that particular instance.
##
## Two example feature-functions, first_last_system_call_feats() and 
## system_call_count_feats(), are defined below.
## The first of these functions indicates what the first and last system-calls 
## made by an executable are, and the second records the total number of system
## calls made by an executable.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.
##
## Computational Caveat
## --------------------
## Because the biggest of any of the xml files is only around 35MB, the code below 
## will parse an entire xml file and store it in memory, compute features, and
## then get rid of it before parsing the next one. Storing the biggest of the files 
## in memory should require at most 200MB or so, which should be no problem for
## reasonably modern laptops. If this is too much, however, you can lower the
## memory requirement by using ElementTree.iterparse(), which does parsing in
## a streaming way. See http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/
## for an example. 

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

import util

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

from tqdm import tqdm


def extract_feats(ffs, direc="train", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target classes, and a list of system-call-history ids in order 
      of their rows in the design matrix.
      
      Note: the vector of target classes returned will contain the true indices of the
      target classes on the training data, but will contain only -1's on the test
      data
    """
    fds = [] # list of feature dicts
    classes = []
    ids = []
    files_lst = os.listdir(direc)
    for datafile in files_lst:
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        rowfd = {}
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        [rowfd.update(ff(tree)) for ff in ffs]
        fds.append(rowfd)

        #print fds
    
    X,feat_dict = make_design_mat(fds,global_feat_dict)
    
    X = X.toarray(); N = X.shape[0]; train_inds = range(0, N / 2); test_inds = range(N / 2, N)
    classes = np.array(classes)

    train_info = [
        X[train_inds], classes[train_inds], [ids[i] for i in train_inds]
    ]

    test_info = [
        X[test_inds], classes[test_inds], [ids[i] for i in test_inds]
    ]

    # return X, feat_dict, np.array(classes), ids
    return train_info, test_info


def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
        
    cols = []
    rows = []
    data = []        
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)
   
    # Will be zero wherever a particular feature did not show up
    # in one feature dict that did in another feature dict (because
    # that feat won't have a row, col pair in cols and rows lists)
    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    return X, feat_dict
    

## Here are two example feature-functions. They each take an xml.etree.ElementTree object, 
# (i.e., the result of parsing an xml file) and returns a dictionary mapping 
# feature-names to numeric values.
## TODO: modify these functions, and/or add new ones.
def first_last_system_call_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if first:
                c["first_call-"+el.tag] = 1
                first = False
            last_call = el.tag  # update last call seen
            
    # finally, mark last call seen
    c["last_call-"+last_call] = 1
    return c

def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
    return c

#returns a dictionary with key as name of system call and value as the frequency of the system call
def frequency(tree):
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c[el.tag] += 1
    return c

def trigrams(tree):
    c = Counter()
    in_all_section = False
    prev_elt = None
    prev_prev_elt = None
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
            prev_elt = None
            prev_prev_elt = None # Reset trigrams for new section
        elif in_all_section:
            if prev_prev_elt == None:
                prev_prev_elt = el.tag
            elif prev_elt == None:
                prev_elt = el.tag
            else:
                c[prev_prev_elt + prev_elt + el.tag] += 1
                prev_prev_elt = prev_elt
                prev_elt = el.tag
    return c

def accuracy(preds, actual):
    diff = preds - actual
    n = len(actual) * 1.0
    return 1 - (np.count_nonzero(diff) / n)

## The following function does the feature extraction, learning, and prediction
def main():
    train_dir = "train"
    test_dir = "test"
    outputfile = "sample_predictions.csv"  # feel free to change this or take it as an argument
    
    # TODO put the names of the feature functions you've defined above in this list
    # ffs = [first_last_system_call_feats, system_call_count_feats, frequency]
    ffs = [trigrams]
    
    # extract features
    print "extracting training features..."
    # X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
    train_info, test_info = extract_feats(ffs, train_dir)
    Xtrain, Ytrain, train_ids = train_info
    Xtest, Ytest, test_ids = test_info
    
    # # Random Forest CV
    # print("Xtrain shape:", Xtrain.shape[1])
    # best_depth = None
    # best_num_features = None
    # best_score = float("-inf")
    # tot_features = Xtrain.shape[1]
    # kfold = KFold(n_splits = 5)
    
    # # On 1000 rows
    # # First CV attempt: 7032 features, optimal num features was 2904.  Depth range 4, 16, 3: best was 13
    # # Second CV attempt: 7032 features, optimal num features was 2600.  Depth range was 10, 30, 5: best was 20
    # #
    # # On all rows
    # # 10326 features, opt num feats was 2800.  Opt depth was 30
    # # 10326 features, opt num feats is 2750, opt depth is 28

    # for depth in range(28, 33, 1):
    #     for max_feat in tqdm(range(2750, 2900, 20)):
    #         kscores = []
    #         for train_ind, test_ind in kfold.split(Xtrain):
    #             xtrain_cv = Xtrain[train_ind]
    #             ytrain_cv = Ytrain[train_ind]

    #             xtest_cv = Xtrain[test_ind]
    #             ytest_cv = Ytrain[test_ind]

    #             rf = RandomForestClassifier(max_depth = depth, max_features = max_feat)
    #             rf.fit(xtrain_cv, ytrain_cv)
    #             preds = rf.predict(xtest_cv)
    #             kscores.append(accuracy(preds, ytest_cv))

    #         score = np.mean(kscores)
    #         if score > best_score:
    #             best_score = score
    #             best_depth = depth
    #             best_num_features = max_feat
    
    # print("Best depth:", best_depth)
    # print("Best features", best_num_features)
    rf = RandomForestClassifier(max_features = 2750, max_depth = 28)
    rf.fit(Xtrain, Ytrain)
    preds = rf.predict(Xtest)
    print("RF:",  accuracy(preds, Ytest))

    rf = RandomForestClassifier()
    rf.fit(Xtrain, Ytrain)
    preds = rf.predict(Xtest)
    print("RF:",  accuracy(preds, Ytest))

    # nb = MultinomialNB(class_prior = [0.0369, 0.0162, 0.012, 0.0103, 0.0133, 0.0126, 0.0172, 0.0133, 0.5214, 0.0068, 0.1756, 0.0104, 0.1218, 0.0191, 0.0130])
    # nb = MultinomialNB(class_prior = [1 for i in range(15)])
    # nb = MultinomialNB(class_prior = [0.5] + [0.2] * 7 + [0.3] + [0.2] + [0.3] + [0.2] + [0.3] + [0.2] * 2) # Weight less common classes higher
    nb = MultinomialNB()
    nb.fit(Xtrain, Ytrain)
    preds = nb.predict(Xtest)
    print("NB:",  accuracy(preds, Ytest))

    
    # # NN CV: based on number neurons in one hidden layer
    # # On 1000 rows, best number of nodes was 281
    # print("Xtrain shape:", Xtrain.shape[1])
    # best_num_nodes = None
    # best_score = float("-inf")
    # N = Xtrain.shape[0]
    # kfold = KFold(n_splits = 5)

    # for num_nodes in tqdm(range(1, (N * 2) // 3, 70)):
    #     kscores = []
    #     for train_ind, test_ind in kfold.split(Xtrain):
    #         xtrain_cv = Xtrain[train_ind]
    #         ytrain_cv = Ytrain[train_ind]

    #         xtest_cv = Xtrain[test_ind]
    #         ytest_cv = Ytrain[test_ind]

    #         nn = MLPClassifier(max_iter = 10000, hidden_layer_sizes = (num_nodes,))
    #         nn.fit(xtrain_cv, ytrain_cv)
    #         preds = nn.predict(xtest_cv)
    #         kscores.append(accuracy(preds, ytest_cv))

    #     score = np.mean(kscores)
    #     if score > best_score:
    #         best_num_nodes = num_nodes
    #         best_score = score

    # print("best num nodes:",  best_num_nodes)

    
    nn = MLPClassifier(max_iter = 10000, hidden_layer_sizes = (281,))
    nn.fit(Xtrain, Ytrain)
    preds = nn.predict(Xtest)
    print("NN:", accuracy(preds, Ytest))


    # print X_train # Not currently a np.array, need to do .toarray()
    # print global_feat_dict

    print "done extracting training features"
    print
    
    # TODO train here, and learn your classification parameters
    print "learning..."
    # learned_W = np.random.random((len(global_feat_dict),len(util.malware_classes)))
    print "done learning"
    print
    
    # # get rid of training data and load test data
    # del X_train
    # del t_train
    # del train_ids
    
    # print "extracting test features..."
    # X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
    # print "done extracting test features"
    # print
    
    # # TODO make predictions on text data and write them out
    # print "making predictions..."
    # preds = np.argmax(X_test.dot(learned_W),axis=1)
    # print "done making predictions"
    # print
    
    # print "writing predictions..."
    # util.write_predictions(preds, test_ids, outputfile)
    # print "done!"

if __name__ == "__main__":
    main()
    
