import random
import sys
reload(sys)
sys.setdefaultencoding('utf8')
file_object  = open(sys.argv[1], "r")

train_file = open(sys.argv[2], "w")
output_file = open(sys.argv[3], "w")

num = int(sys.argv[4])
i = 0;
selected = []
for line in file_object:
    if (i < num): 
        selected.append(line);
        i += 1
    else:
        a = random.randint(1, i);
        if a <= num:
            b = random.randint(0,num-1)
            out_sen = selected[b]
            selected[b] = line;
        else:
            out_sen = line;
        train_file.write(out_sen.encode("utf-8"))
            
for sen in selected:
    output_file.write(sen.encode("utf-8"))         
