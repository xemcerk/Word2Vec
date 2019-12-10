from gensim.models import Word2Vec

if __name__ == '__main__':
    testWords = ['苹果', '华为', '谷歌']
    model = Word2Vec.load('./model/wiki_zh_word2vec.model')
    for word in testWords:
        result = model.most_similar(word);
        print(word)
        print(result)
