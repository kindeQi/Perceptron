from pathlib import Path
from collections import defaultdict
import os
from math import log, log2
import sys, glob
import json
import random
from random import shuffle
import numpy as np

class bayes_dataset(object):
    def __init__(self, dataset):
        '''
        Arguments:
        dataset: List[List[List[str(Path), tuple(label1, label2)]]], shape(4, k)
        '''
        # self.dataset = [dataset[i][j] for i in range(len(dataset)) for j in range(len(dataset[0]))]
        self.dataset = dataset
        self.captial_list = set([chr(i) for i in range(65, 90 + 1)])
        self.lower_list   = set([chr(i) for i in range(97, 122 + 1)])
        
    def __getitem__(self, index):
        '''
        Arguments:
        index: int, the inde of required item
        
        Description:
        get the data, tokenize and get the target
        '''
        path, label = self.dataset[index]
        return self.tokenize(path, label), label
    
    def __len__(self):
        return len(self.dataset)
    
    def tokenize(self, file_path, file_label):
        '''
        Argument:
        file_path: the path of input file, Path()
        file_label: List(label1, label2)
        
        Description:
        remove all special character, lowercase the word

        TODO
        use regular expression
        '''
        content = ""
        with open(file_path, 'r') as f:
            original_content = f.readline()
            content = original_content[:-1]
            content = content.split(" ")
            
            word_list = []
            for word in content:
                w = ''
                for letter in word:
                    if letter in self.captial_list:
                        w += str.lower(letter)
                    elif letter in self.lower_list:
                        w += letter
                    elif letter in set(['!', '?', '/', '&', '*']):
                        w += letter
                if w != '':
                    word_list.append(w)
            
        return word_list

class F1_score(object):
    def __init__(self, num_class):
        
        self.num_class = num_class
        
        self.true_positive = 'true_positive'
        self.true_negative = 'true_negative'
        self.false_positive = 'false_positive'
        self.false_negative = 'false_negative'
        
        self.class_score = [defaultdict(int) for _ in range(self.num_class)]
    
    def add_score(self, gold_cls, sys_cls):
        '''
        Arguments
        gold_cls: int, [0, 3]
        sys_cls: int, [0, 3]
        '''
        if gold_cls == sys_cls:
            self.class_score[gold_cls][self.true_positive] += 1
        else:
            self.class_score[gold_cls][self.false_negative] += 1
            self.class_score[sys_cls][self.false_positive] += 1
    
    def calculate_F1_score(self):
        recall = [c[self.true_positive] / (c[self.true_positive] + c[self.false_negative] + epsilon) for c in self.class_score]
        precision = [c[self.true_positive] / (c[self.true_positive] + c[self.false_positive] + epsilon) for c in self.class_score]
        
        F1 = [(2 * recall[i] * precision[i]) / (recall[i] + precision[i] + epsilon) for i in range(self.num_class)]
        
        return sum(F1) / len(F1), recall, precision, F1

class Vectorize():
    def __init__(self, data_loader, norm='none'):
        '''
        Arguments:
        Data_loader: bayes_dataset
        Norm: ['none', 'l1', 'l2']. how to normalize the tf-idf vector

        Description:
        Convert the document into tf-idf vector
        '''
        self.dataloader = data_loader
        self.wf_per_doc = []
        self.df_per_word = defaultdict(int)
        self.vocabulary = set()
        self.num_word_per_doc = []
        if norm == 'none':
            self.norm = self.no_norm
        elif norm == 'l1':
            self.norm = self.l1_norm
        elif norm == 'l2':
            self.norm = self.l2_norm
        
        self.word_count()
        self.num_documents = len(data_loader)

        self.sorted_vocabulary = sorted(self.vocabulary)
        self.word_pos = defaultdict(int)
        for i, w in enumerate(self.sorted_vocabulary):
            self.word_pos[w] = i
        
    def word_count(self):
        for idx in range(len(self.dataloader)):
            wf = defaultdict(int)
            content, label = self.dataloader[idx]
            used = set()
            
            self.num_word_per_doc.append(len(content))
            
            for w in content:
                
                # df
                if w not in used:
                    used.add(w)
                    self.df_per_word[w] += 1
                
                # vocabulary, wf
                self.vocabulary.add(w)
                wf[w] += 1
            self.wf_per_doc.append(wf)
    
    def tf_idf(self, idx):
        '''
        Description:
        get the tf_idf vector of a document, not all documents, in case its too large for memory
        
        Arguemnt:
        idx: index of docuemnt, [0, len(self.dataloader) - 1]
        '''
        
        res = np.zeros(len(self.sorted_vocabulary))
        for w in self.wf_per_doc[idx].keys():  
            if w not in self.vocabulary:
                continue
                
            pos = self.word_pos[w]
            tf = self.wf_per_doc[idx][w]
            
            idf = 1.0 + log((1 + self.num_documents) / (1 + self.df_per_word[w]))
            res[pos] += tf * idf
        return self.norm(res)

    def l1_norm(self, X):
        return X / np.sum(X)
    
    def l2_norm(self, X):
        return X / np.sum(X ** 2) ** 0.5

    def no_norm(self, X):
        return X

def transform_label(data_loader, label_id):
    label_transform = {0: -1, 1: 1, 2: -1, 3:1}
    res = []
    for idx in range(len(data_loader)):
        if label_id == 1:
            y = data_loader[idx][1][0]
        elif label_id == 2:
            y = data_loader[idx][1][1]
        y = label_transform[y]
        res.append(y)
    return res

def train(max_iter, weight, bias, data_loader, X, Y):
    for idx in range(max_iter):
        idx = idx % len(data_loader)
        
        x = X[idx]
        y = Y[idx]
        a = np.dot(weight, x) + bias

        if y * a <= 0:
            weight += y * x
            bias += y
    return weight, bias

def train_average(max_iter, weight, bias, data_loader, X, Y):
    sum_weight = np.zeros(len(weight))
    sum_bias = np.zeros(len(bias))

    c = 0
    for idx in range(max_iter):
        idx = idx % len(data_loader)
        
        x = X[idx]
        y = Y[idx]
        a = np.dot(weight, x) + bias

        if y * a <= 0:
            sum_weight += c * weight
            sum_bias += c * bias

            c = 0
            weight += y * x
            bias += y
        else:
            c += 1

    return sum_weight, sum_bias


def write_json(file_name, weight_label1, bias_label1, weight_label2, bias_label2, vocabulary, num_documents, df_per_word):
    json_object = {
        'weight_label1':weight_label1.tolist(), 'bias_label1':bias_label1.tolist(),
        'weight_label2':weight_label2.tolist(), 'bias_label2':bias_label2.tolist(),
        'vocabulary':list(vocabulary),'num_documents':num_documents, 'df_per_word':df_per_word
        }

    with open(file_name, 'w') as f:
        json.dump(json_object, f)

def l2_norm(input_X):
    for idx in range(len(input_X)):
        input_X[idx] = input_X[idx] / np.sum(input_X[idx] ** 2) ** 0.5
    return input_X

def write_vanilla_parameters(max_iter1, max_iter2, trn_data_loader, trn_vectorize, trn_X, trn_label1, trn_label2):

    weight_label1, bias_label1 = np.zeros(len(trn_vectorize.vocabulary)), np.zeros(1) + 1
    weight_label2, bias_label2 = np.zeros(len(trn_vectorize.vocabulary)), np.zeros(1) + 1

    weight_label1, bias_label1 = train(max_iter1, weight_label1, bias_label1, trn_data_loader, trn_X, trn_label1)
    weight_label2, bias_label2 = train(max_iter2, weight_label2, bias_label2, trn_data_loader, trn_X, trn_label2)

    write_json('vanillamodel.txt', weight_label1, bias_label1,
    weight_label2, bias_label2, trn_vectorize.vocabulary,
    trn_vectorize.num_documents, trn_vectorize.df_per_word)

def write_average_parameters(max_iter1, max_iter2, trn_data_loader, trn_vectorize, trn_X, trn_label1, trn_label2):

    weight_label1, bias_label1 = np.zeros(len(trn_vectorize.vocabulary)), np.zeros(1) + 1
    weight_label2, bias_label2 = np.zeros(len(trn_vectorize.vocabulary)), np.zeros(1) + 1

    weight_label1, bias_label1 = train_average(max_iter1, weight_label1, bias_label1, trn_data_loader, trn_X, trn_label1)
    weight_label2, bias_label2 = train_average(max_iter2, weight_label2, bias_label2, trn_data_loader, trn_X, trn_label2)

    write_json('averagedmodel.txt', weight_label1, bias_label1,
    weight_label2, bias_label2, trn_vectorize.vocabulary,
    trn_vectorize.num_documents, trn_vectorize.df_per_word)


if __name__ == "__main__":
    
    # 1. build the dataset
    random.seed(1)
    epsilon = 1e-8

    split_word = '\\' if sys.platform == 'win32' else '/'
    path_to_input = './op_spam_v1.4' if sys.platform == 'win32' else str(sys.argv[1])

    all_files = glob.glob(os.path.join(path_to_input,'*/*/*/*.txt'))

    idx_catagory = {0: 'negative', 1: 'positive', 2: 'deceptive', 3: 'truthful'}
    catagory_idx = {'negative': 0, 'positive': 1, 'deceptive': 2, 'truthful': 3}

    trn_dataset = []
    val_dataset = []

    
    for file in all_files:
        class1, class2, fold, fname = file.split(split_word)[-4:]
        class1, class2 = class1.split('_')[0], class2.split('_')[0]
        class1, class2 = catagory_idx[class1], catagory_idx[class2]
        if fold != 'fold5':
            trn_dataset.append([file, (class1, class2)])
            
    # 2. shuffle the dataset
    shuffle(trn_dataset)


    max_iter1, max_iter2 = 4200, 4200
    trn_data_loader = bayes_dataset(trn_dataset)

    trn_vectorize = Vectorize(trn_data_loader)
    trn_X = [trn_vectorize.tf_idf(idx) for idx in range(len(trn_data_loader))]

    trn_label1, trn_label2 = transform_label(trn_data_loader, 1), transform_label(trn_data_loader, 2)

    write_vanilla_parameters(max_iter1, max_iter2, trn_data_loader, trn_vectorize, trn_X, trn_label1, trn_label2)
    trn_X = l2_norm(trn_X)
    write_average_parameters(max_iter1, max_iter2, trn_data_loader, trn_vectorize, trn_X, trn_label1, trn_label2)