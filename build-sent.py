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
            text = re.sub(r'[\u02BB\u02BC\u066C\u2018-\u201A\u275B\u275C]','\'', text)
            text = re.sub(r'[\u0027\u02B9\u02BB\u02BC\u02BE\u02C8\u02EE\u0301\u0313\u0315\u055A\u05F3\u07F4\u07F5\u1FBF\u2018\u2019\u2032\uA78C\uFF07]', '\'', text)
            text = re.sub('[\u201C-\u201E\u2033\u275D\u275E\u301D\u301E]', r'', text)
            text = ''.join(ch for ch in text if ord(ch) < 128)
            text = ''.join(ch for ch in text if ord(ch) != ord(sep)) # care with *.csv files
            # course specific
            text = re.sub(r'Dr.', r'Dr', text)
            text = re.sub(r'Prof.', r'Prof', text)
            text = re.sub(r'Drs.', r'Drs', text)
            text = re.sub(r'Profs.', r'Profs', text)
            text = re.sub(r'U.S.', r'US', text)
            text = re.sub(r'etc.', r'etc', text)
            text = re.sub(r'E\.', r'E', text)
            text = text.lower()
            # writing
            topic, st, num = files[i].split('-')
            idx = num.find('.')
            num = num[:idx]
            sentences = text.split('.') # periods
            for sentence in sentences:
                if drop_punct:
                    sentence = re.sub(r'[^\w\s\']', r' ', sentence)
                    sentence = re.sub(r'_', r' ', sentence)
                output.write(topic)
                output.write(sep)
                output.write(st)
                output.write(sep)
                output.write(num)
                output.write(sep)
                output.write(sentence)
                output.write('\n')