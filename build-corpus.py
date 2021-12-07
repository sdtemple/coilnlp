# modules
import sys
import os
import re

# inputs
folder = sys.argv[1]
sep = sys.argv[2]
drop_punct = int(sys.argv[3])
name = sys.argv[4]

# files in folder
files = os.listdir(folder)
files = [file for file in files if file[-4:] == '.txt']

with open(name, 'w') as output:
    # header
    output.write('TOPIC,STATE,DOCNUM,TEXT\n')
    for i in range(len(files)):
        # debugging
        print(files[i])
        with open(folder + files[i], 'r') as f:
            # preprocessing
            text = f.read()
            text = re.sub(r'\n', r' ', text)
            text = ''.join(ch for ch in text if ord(ch) < 128)
            text = ''.join(ch for ch in text if ord(ch) != ord(sep))
            if drop_punct:
                text = re.sub(r'[^\w\s]', r' ', text)
                text = re.sub(r'_', r' ', text)
            text = text.lower()
            # writing
            topic, st, num = files[i].split('-')
            idx = num.find('.')
            num = num[:idx]
            output.write(topic)
            output.write(sep)
            output.write(st)
            output.write(sep)
            output.write(num)
            output.write(sep)
            output.write(text)
            output.write('\n')