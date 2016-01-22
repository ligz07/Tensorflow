import numpy as np
import re
import itertools
from collections import Counter
import codecs
import gensim
import tensorflow as tf
def singleton(cls, *args, **kw):  
    instances = {}  
    def _singleton():  
        if cls not in instances:  
            instances[cls] = cls(*args, **kw)  
        return instances[cls]  
    return _singleton  
 
@singleton  
class MyClass4(object):  
    model = None
    def __init__(self):  
        self.model =  gensim.models.Word2Vec.load("/home/guanglei/word2vec/wiki.zh.text.model");

def get_word_vector(key):
    class1 = MyClass4();
    model = class1.model;
    try:

        val = model[key]
    except:
        val = None
    return val
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
##    positive_examples = list(open("./data/rt-polarity.pos").readlines())
#    positive_examples = [s.strip() for s in positive_examples]
#    negative_examples = list(open("./data/rt-polarity.neg").readlines())
#    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = [];
    yp_list = [];
    dict_label = {};
    lable_inv = [];
    file_object = codecs.open("./data/fortest.txt","r", "utf-8").readlines();
    for line in file_object:
        arr = line.split("\t");
        if len(arr) < 2:
            print line;
            break;
        sent = arr[0].strip();
        label = clean_str(arr[1]);
        x_text.append(sent);
        if not dict_label.has_key(label):
            dict_label[label] = len(dict_label);
            lable_inv.append(label);

        yp_list.append(dict_label[label]);
	
    #x_text = positive_examples + negative_examples
    #x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    #positive_labels = [[0, 1] for _ in positive_examples]
    #negative_labels = [[1, 0] for _ in negative_examples]
    #y = np.concatenate([positive_labels, negative_labels], 0)
    y = []
    for la in yp_list:
        ve = np.zeros(len(dict_label));
        ve[la] = 1; 
        y.append(ve);
    return [x_text, y, lable_inv]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, lable_inv = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    #sentences_padded = sentences;
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    vv = np.random.uniform(-1.0,1.0, len(vocabulary)*400);
    W = np.float32(vv.reshape(len(vocabulary), 400));
    
    for i in vocabulary.keys():
        v = get_word_vector(i);
        if v is not None:
            W[vocabulary[i]] = v;
    return [x, y, vocabulary, vocabulary_inv, W, lable_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
#f = open("./out", "r");
pp = load_data()
print pp[4]
#for sen in pp[0]:
#    for w in sen:
#        print get_word_vector(w);
