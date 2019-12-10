from gensim.models import Word2Vec
import subprocess
import pandas as pd

model = Word2Vec.load('./model/wiki_zh_word2vec.model')

WORD_PAIRS_FILEPATH = 'pku_sim_test.txt'
N_WORD_PAIRS = subprocess.check_output(
    'wc -l {}'.format(WORD_PAIRS_FILEPATH), shell=True)
N_WORD_PAIRS = int(N_WORD_PAIRS.split()[0])

word_pairs = []
with open('pku_sim_test.txt', 'r') as f:
    for i in range(N_WORD_PAIRS):
        line = f.readline()
        line = line.strip('\n')
        if(i == 0):
            line = line.strip('\ufeff')
        word_pair = line.split('\t', 2)
        word_pairs.append(word_pair)

sim_list = []
word_list_a = []
word_list_b = []
for i in range(N_WORD_PAIRS):
    wa = word_pairs[i][0]
    wb = word_pairs[i][1]
    word_list_a.append(wa)
    word_list_b.append(wb)
    if(wa not in model.wv.vocab or wb not in model.wv.vocab):
        sim_list.append('OOV')
    else:
        sim = model.wv.similarity(wa, wb)
        sim_list.append(sim)

data = {'word_list_a': word_list_a,
        'word_list_b': word_list_b, 'similarity': sim_list}
df = pd.DataFrame(data)
df.to_csv('result.txt', header=None, index=None, sep='\t', mode='a')
