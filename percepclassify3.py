from pathlib import Path
from collections import defaultdict
import os
from math import log, log2
import sys, glob
import json
import numpy as np
epsilon = 1e-8


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
        path = self.dataset[index]
        return self.tokenize(path), path
    
    def __len__(self):
        return len(self.dataset)
    
    def tokenize(self, file_path):
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

def validation(weight, bias, data_loader, X, Y, f1):
    '''
    Arguments:
    weight: ndarray, the weight of perceptron
    bias: int, the bias of perceptron
    dataloader: bayes_dataset
    X: ndarray, input vector
    Y: ground truth
    f1: F1_score
    '''
    for idx in range(len(data_loader)):
        x = X[idx]
        y = Y[idx]
        a = np.dot(weight, x) + bias
        
        pred = 1 if a >= 0 else 0
        gt = 1 if y >=0 else 0
        f1.add_score(gt, pred)
    
    res, recall, precision, F1 = f1.calculate_F1_score()
    print('Average F1: {}\n Recall: {}\n Precision: {}\n, F1: {}'.format(res, recall, precision, F1))

if __name__ == "__main__":
    
    # 1. load the weights and the model file
    split_word = '\\' if sys.platform == 'win32' else '/'
    # model_file = './averagedmodel.txt' if sys.platform == 'win32' else str(sys.argv[1])
    # path_to_input = './op_spam_v1.4' if sys.platform == 'win32' else str(sys.argv[2])
    model_file = str(sys.argv[1])
    path_to_input = './op_spam_v1.4'
    all_files = glob.glob(os.path.join(path_to_input,'*/*/*/*.txt'))
    output_file = 'percepoutput.txt'

    parameters = None
    with open(model_file, 'r') as f:
        parameters = json.load(f)

    weight_label1, bias_label1, weight_label2, bias_label2, vocabulary, num_documents, df_per_word = parameters['weight_label1'], parameters['bias_label1'], parameters['weight_label2'], parameters['bias_label2'], parameters['vocabulary'], parameters['num_documents'], parameters['df_per_word']

    idx_catagory = {0: 'negative', 1: 'positive', 2: 'deceptive', 3: 'truthful'}
    catagory_idx = {'negative': 0, 'positive': 1, 'deceptive': 2, 'truthful': 3}

    weight_label1 = np.array(weight_label1)
    weight_label2 = np.array(weight_label2)
    bias_label1 = np.array(bias_label1)
    bias_label2 = np.array(bias_label2)
    vocabulary = vocabulary
    df_per_word = defaultdict(int, df_per_word)

    test_dataset = []
    for file in all_files:
        class1, class2, fold, fname = file.split(split_word)[-4:]
        if fold != 'fold5':
            # trn_dataset.append([file, (class1, class2)])
            pass
        else:
            test_dataset.append(file)

    data_loader = bayes_dataset(test_dataset)
    # data_loader = [data_loader[idx] for idx in range(len(data_loader))]

    if 'vanillamodel.txt' in model_file:
        test_vectorize = Vectorize(data_loader, 'none')
    else:
        test_vectorize = Vectorize(data_loader, 'l2')

    test_vectorize.vocabulary = vocabulary
    test_vectorize.num_documents = num_documents
    test_vectorize.df_per_word = df_per_word
    test_vectorize.sorted_vocabulary = sorted(test_vectorize.vocabulary)
    for i, w in enumerate(test_vectorize.sorted_vocabulary):
        test_vectorize.word_pos[w] = i

    test_X = [test_vectorize.tf_idf(idx) for idx in range(len(data_loader))]

    f1 = F1_score(4)
    output_list = []
    for idx, (content, label) in enumerate(data_loader):
        x = test_X[idx]
        y1 = np.dot(weight_label1, x) + bias_label1
        y2 = np.dot(weight_label2, x) + bias_label2

        pred1 = 0 if y1 < 0 else 1
        pred2 = 1 if y2 < 0 else 0

        class1, class2, fold, fname = label.split(split_word)[-4:]
        class1, class2 = class1.split('_')[0], class2.split('_')[0]
        class1, class2 = catagory_idx[class1], catagory_idx[class2]

        l1 = 'negative' if y1 < 0 else 'positive'
        l2 = 'deceptive' if y2 < 0 else 'truthful'
        f1.add_score(class1, catagory_idx[l1])
        f1.add_score(class2, catagory_idx[l2])

    res, recall, precision, f1 = f1.calculate_F1_score()
    print('Average F1: {}'.format(res))
    #     l1 = 'negative' if y1 < 0 else 'positive'
    #     l2 = 'deceptive' if y2 < 0 else 'truthful'
    #     output_list.append('{} {} {}\n'.format(l2, l1, label))

    # with open(output_file, 'w') as f:
    #     f.writelines(output_list)




