# 算法描述

主要通过三个部分实现SGNS

1. 数据预处理
2. 模型训练
3. 模型应用

## 数据预处理

预处理主要对wiki的中文语料库进行繁简转化/分词

### 1-语料库解压缩转换为wiki.zh.text（process_wiki.py）

```bash
#执行语句
python process_wiki.py ./data/zhwiki-latest-pages-articles.xml.bz2 ./data/wiki.zh.text
```

### 2-繁简转换

```bash
#通过opencc进行繁简转换得到简体语料库 wiki.zh.jian.text
opencc -i ./data/wiki.zh.text -o ./data/wiki.zh.jian.text -c t2s.json
```

### 3-分词（word_segmentation.py）

```bash
#执行语句
python word_segmentation.py
```

详细代码：

```python
#采用jieba分词将简体文件转换为train.txt训练集
for line in open("./data/wiki.zh.jian.text"):
        for i in re.sub('[a-zA-Z0-9]', '', line).split(' '):
            if i != '':
                data = list(jieba.cut(i, cut_all=False))
                readline = ' '.join(data) + '\n'
                f.write(readline)
```



## 模型训练

```bash
#执行语句
python train.py ./data/train.txt ./model/wiki_zh_word2vec.model ./model/wiki_zh_vectors.txt
#其中train 为输入数据 word2vec vectors两个分别是模型结果
```



```python
#采用gensim库进行训练
Word2Vec(LineSentence(input), size=100,sg=1, window=2, min_count=5,hs=0, workers=multiprocessing.cpu_count())
#Word2Vec模型训练函数size维数100 window窗口大小2 sg=1采用Skip-Gram模型 hs=0采用NegativeSimple
```

**参数含义：**

**sentences:** 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。后面我们会有从文件读出的例子。

**size:** 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。

**window:** 即词向量上下文最大距离，这个参数在我们的算法原理篇中标记为c，window越大，则和某一词较远的词也会产生上下文关系。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间。

**sg:** 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。

**hs:** 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。
negative:即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。

**cbow_mean:** 仅用于CBOW在做投影的时候，为0，则算法中的xw为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示xw,默认值也是1,不推荐修改默认值。

**min_count**:需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。

**iter:** 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。

**alpha:** 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为η，默认是0.025。

**min_alpha:** 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。

## 模型应用

```bash
#执行命令
python test_sim_cal.py
```



```python
#关键代码
model = Word2Vec.load('./model/wiki_zh_word2vec.model')#通过gensim库中api读取模型
sim = model.wv.similarity(wa, wb)#通过封装好的模型方法计算相似度
```

结果存放在result.txt中