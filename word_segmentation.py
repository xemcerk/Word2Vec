import codecs, sys, re
import jieba

if __name__ == '__main__':
    f = codecs.open('./data/train.txt', "a+", 'utf-8')
    for line in open("./data/wiki.zh.jian.text"):
        for i in re.sub('[a-zA-Z0-9]', '', line).split(' '):
            if i != '':
                data = list(jieba.cut(i, cut_all=False))
                readline = ' '.join(data) + '\n'
                f.write(readline)
    f.close()
    print('Finished zh segmentation')