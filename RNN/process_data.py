import jieba
import codecs
labled_object = open("data/labled","r")
#original_object = open("data/original","r")
output_file = open("data/output", "w")

for line in labled_object:
    seg_list = jieba.cut(line, cut_all = False);
    list = []
    sentence = [];
    for i, word in enumerate(seg_list):
        w = word.strip();
        arr = w.split("*")
        if len(arr) == 1:
            list.Append(0);
            sentence.Append(w);
        else:
            if len(arr[0]) == 0:
                list[-1] = 1;
            else:
                sentence.Append(arr[0])
                list.Append(1)

            if not len(arr[1]) == 0:
                sentence.Append(arr[1])

    output_file.write(" ".join(sentence).encode("utf-8") + "\t" + " ".join(list));