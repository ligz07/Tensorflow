import codecs
import numpy as np
from collections import Counter
import itertools
import gensim
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
#    return None
    class1 = MyClass4();
    model = class1.model;
    try:
        val = model[key]
    except:
        val = None
    return val

def pad_sentences(sentences, labels, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    padded_labels = []
    padded_weights= []
    for i in range(len(sentences)):
        sentence = sentences[i]
        label = labels[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        new_labels = label + [0] * num_padding
        weight = [1] * len(sentence) + [0] * num_padding
        padded_sentences.append(new_sentence)
        padded_labels.append(new_labels);
        padded_weights.append(weight);
    return padded_sentences, padded_labels, padded_weights

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

def load_data_and_labels(file_name):
    x_text = [];
    y_list = [];
    dict_label = {};
    lable_inv = [];
    sent = []
    lables = []
    file_object = codecs.open(file_name,"r", "utf-8")
    for line in file_object:
        arr = line.split("\t");
        if len(arr) < 2:
            print line;
            break;
        sent.append(arr[0].strip());
        lables.append(arr[1].strip());

    x_text = [s.split(" ") for s in sent]
    y_list = [s.split(" ") for s in lables]

    return x_text, y_list

def load_data():
    x, y = load_data_and_labels("data/train")
    dev_x, dev_y = load_data_and_labels("data/test")
    dev_num = len(dev_x);
    x += dev_x;
    y += dev_y;
    pad_sentence, pad_labels, pad_weights = pad_sentences(x, y);

    voc, voc_inv = build_vocab(pad_sentence);
    """ 
     y_list = []
    for la in pad_labels:
        p = []
        for l in la:
            ve = [0]*2;
            ve[int(l)] = 1;
            p.append(ve)
        y_list.append(p);
    """
    x = np.array([[voc[word] for word in sentence] for sentence in pad_sentence])
    y = np.array([[int(l) for l in sentence ]for sentence in pad_labels])
    vv = np.random.uniform(-1.0,1.0, len(voc)*400);
    W = np.float32(vv.reshape(len(voc), 400));

    for i in voc.keys():
        v = get_word_vector(i);
        if v is not None:
            W[voc[i]] = v;
    dev_x = x[-dev_num:]
    dev_y = y[-dev_num:]
    x = x[:-dev_num];
    y = y[:-dev_num];
    dev_pad = pad_weights[-dev_num:]
    pad_weights = pad_weights[:-dev_num]
    return [x, y, voc, voc_inv, W, pad_weights, dev_x, dev_y, dev_pad]

#print dx
#for s in dx:
#    for k in s:
#        print voc_inv[k].encode("utf-8");
'''for key in voc_inv:
    print key.encode("utf-8");

print "------"

for p in x:
    print x

print "-----------"

for k in w:
    print k;'''

