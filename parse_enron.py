import os

path = '../enron-email'

files = []
for i in os.listdir(path):
    path_ = path + '/' + i + '/all_documents'
    if os.path.isdir(path_):
        for j in os.listdir(path_):
            files.append(path_ + '/' + j)

print(len(files), 'files')

text = []
for file in files:
    with open(file, 'r', errors='ignore') as f:
        text.append(f.read())

all_text = '\n'.join(text)

with open('tests/enron-email.txt', 'w') as f:
    f.write(all_text)