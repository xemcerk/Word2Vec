1. **install package**
    
        pip install -r requirements.txt

2. **download data**

        . download raw data from this link:https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
        . create data file at project root
        . put raw data to data file
             
3. **process corpus, form xml to text**

        python process_wiki.py ./data/zhwiki-latest-pages-articles.xml.bz2 ./data/wiki.zh.text
4. **process corpus, form tranditional to simplified**

        opencc -i ./data/wiki.zh.text -o ./data/wiki.zh.jian.text -c t2s.json
        
5. **process corpus, sub En and segmentation zh.jian**

        python word_segmentation.py
        
6. **train**

        python train.py ./data/train.txt ./model/wiki_zh_word2vec.model ./model/wiki_zh_vectors.txt