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
    class1 = MyClass4();
    model = class1.model;
    try:
        val = model[key]
    except:
        val = None
    return val

def load_data_and_labels():