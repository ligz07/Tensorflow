import codecs
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
        new_labels = label + [0] * new_padding
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

    return x_text, y_list

def load_data():
    x, y = load_data_and_labels()
    pad_sentence, pad_labels = pan_senteces(x);

    voc, voc_inv = build_vocab(pad_sentence);

    x = np.array([[voc[word] for word in sentence] for sentence in pad_sentence])
    y = np.array(labels)
    vv = np.random.uniform(-1.0,1.0, len(vocabulary)*400);
    W = np.float32(vv.reshape(len(vocabulary), 400));

    for i in vocabulary.keys():
        v = get_word_vector(i);
        if v is not None:
            W[vocabulary[i]] = v;
    return [x, y, vocabulary, vocabulary_inv, W]

x,y,voc, voc_inv, w = load_data();
