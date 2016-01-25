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
    return None
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
    for i in range(len(sentences)):
        sentence = sentences[i]
        label = labels[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        new_labels = label + [0] * num_padding
        padded_sentences.append(new_sentence)
        padded_labels.append(new_labels);
    return padded_sentences, padded_labels

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

def load_data_and_labels():
    x_text = [];
    y_list = [];
    dict_label = {};
    lable_inv = [];
    sent = []
    lables = []
    file_object = codecs.open("data/output","r", "utf-8").readlines();
    for line in file_object:
        arr = line.split("\t");
        if len(arr) < 2:
            print line;
            break;
        sent.append(arr[0].strip());
        lables.append(arr[1].strip());

    x_text = [s.split(" ") for s in sent]
    y_list = [s.split(" ") for s in lables]

    y = []
    for la in y_list:
        p = []
        for l in la:
            ve = np.zeros(2);
            ve[int(l)] = 1;
            p.append(ve)
        y.append(p);
    return x_text, y

def load_data():
    x, y = load_data_and_labels()
    pad_sentence, pad_labels = pad_sentences(x, y);

    voc, voc_inv = build_vocab(pad_sentence);

    x = np.array([[voc[word] for word in sentence] for sentence in pad_sentence])
    y = np.array(pad_labels)
    vv = np.random.uniform(-1.0,1.0, len(voc)*400);
    W = np.float32(vv.reshape(len(voc), 400));

    for i in voc.keys():
        v = get_word_vector(i);
        if v is not None:
            W[voc[i]] = v;
    return [x, y, voc, voc_inv, W]

#x,y,voc, voc_inv, w = load_data();
'''for key in voc_inv:
    print key.encode("utf-8");

print "------"

for p in x:
    print x

print "-----------"

for k in w:
    print k;'''

