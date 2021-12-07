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
    output.write('TOPIC,STATE,DOCNUM,SENTENCE\n')
    for i in range(len(files)):
        # debugging
        print(files[i])
        with open(folder + files[i], 'r') as f:
            # preprocessing
            text = f.read()
            text = re.sub(r'\n', r' ', text)
            text = re.sub(r'\?', r'.', text)
            text = re.sub(r'!', r'.', text)
            text = ''.join(ch for ch in text if ord(ch) < 128)
            text = ''.join(ch for ch in text if ord(ch) != ord(sep))
            # course specific
            text = re.sub(r'Dr.', r'Dr', text)
            text = re.sub(r'Prof.', r'Prof', text)
            text = re.sub(r'Drs.', r'Drs', text)
            text = re.sub(r'Profs.', r'Profs', text)
            text = re.sub(r'U.S.', r'US', text)
            text = text.lower()
            # writing
            topic, st, num = files[i].split('-')
            idx = num.find('.')
            num = num[:idx]
            sentences = text.split('.') # periods
            for sentence in sentences:
                if drop_punct:
                    sentence = re.sub(r'[^\w\s]', r' ', sentence)
                    sentence = re.sub(r'_', r' ', sentence)
                output.write(topic)
                output.write(sep)
                output.write(st)
                output.write(sep)
                output.write(num)
                output.write(sep)
                output.write(sentence)
                output.write('\n')