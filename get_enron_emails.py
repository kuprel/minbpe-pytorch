import os
import pathlib

path_raw = 'tests/enron_emails'
path_txt = 'tests/enron.txt'
data_url = 'https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz'

os.system(f'wget {data_url} -O enron.tar.gz')
print('Extracting...')
os.system('tar -xf enron.tar.gz')
os.remove('enron.tar.gz')
os.rename('maildir', path_raw)

files = [str(i) for i in pathlib.Path(path_raw).rglob("*/all_documents/*.")]

print(len(files), 'files')

text = ""
for file in files:
    with open(file, 'r') as f:
        text += f.read() + '\n'

with open(path_txt, 'w') as f:
    f.write(text)

print('Wrote to', path_txt)