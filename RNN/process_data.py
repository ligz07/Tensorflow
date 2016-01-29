import jieba
import sys
input_file = sys.argv[1]
labled_object = open(input_file,"r")
#original_object = open("data/original","r")
output_file = open("data/new_output", "w")
output_file1 = open("data/sort_output", "w")
whole_sentences = {}
for line in labled_object:
    seg_list = jieba.cut(line, cut_all = False);
    list = []
    sentence = [];
    for i, word in enumerate(seg_list):
        w = word.strip();
        arr = w.split("*")
        if len(arr) == 1:
            if len(w) > 0:
                list.append("0");
                sentence.append(w);
        else:
            if len(arr[0]) == 0:
                list[-1] = "1";
            else:
                sentence.append(arr[0])
                list.append("1")

            if not len(arr[1]) == 0:
                sentence.append(arr[1])

    assert(len(sentence) == len(list));
    output_file.write(" ".join(sentence).encode("utf-8") + "\t" + " ".join(list) + "\n");
    
    key = len(sentence);
    if not whole_sentences.has_key(key): 
        whole_sentences[key] = [];
    l = whole_sentences[key] 
    l.append((sentence, list));

for k, v in whole_sentences.items():
    for l in v:
        output_file1.write(" ".join(l[0]).encode("utf-8") + "\t" + " ".join(l[1]) + "\n");
