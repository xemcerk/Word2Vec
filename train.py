import os, sys, logging, multiprocessing
from gensim.corpora import wikicorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(("Running %s" % ' '.join(sys.argv)))

    if (len(sys.argv) < 4):
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    input, output1, output2 = sys.argv[1:4]
    model = Word2Vec(LineSentence(input), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(output1)
    model.wv.save_word2vec_format(output2, binary=False)
