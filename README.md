# awesome-ukrainian-nlp
Curated list of Ukrainian natural language processing (NLP) resources (corpora, pretrained models, libriaries, etc.)


## 1. Datasets / Corpora

### Monolingual

* [OSCAR](https://oscar-corpus.com/) — shuffled sentences extracted from [Common Crawl](https://commoncrawl.org/) and classified with a language detection model. 
* [Brown-UK](https://github.com/brown-uk/corpus) — carefully curated corpus of modern Ukrainian language
* [UberText](https://lang.org.ua/uk/corpora/#anchor4) — 6 GB of news, Wikipedia and fiction texts
* [Wikipedia](https://dumps.wikimedia.org/ukwiki/latest/)

### Labeled

* [NER-uk](https://github.com/lang-uk/ner-uk) — Brown-UK labeled for named entities


### Dictionaries

* [ВЕСУМ](https://github.com/brown-uk/dict_uk) — POS tag dictionary. Can generate a list of all wordforms valid for spelling.
* [Tonal dictionary](https://github.com/lang-uk/tone-dict-uk)




## 2. Tools

* [tree_stem](https://github.com/amakukha/stemmers_ukrainian) — stemmer
* [pymorphy2](https://github.com/kmike/pymorphy2) + [pymorphy2-dicts-uk](https://pypi.org/project/pymorphy2-dicts-uk/) — POS tagger and lemmatizer
* [LanguageTool](https://languagetool.org/uk/) -- grammar, style and spell checker

 

## 3. Pretrained models

### Language models

* [RoBERTa](https://huggingface.co/youscan/ukr-roberta-base)

### Machine translation

* [Helsinki NLP models](https://huggingface.co/Helsinki-NLP) — 10 language pairs:
  - Ukrainian-English
  - Ukrainian-Finnish
  - Ukrainian-French
  - Ukrainian-Spanish
  - Ukrainian-Swedish
  - English-Ukrainian
  - Finnish-Ukrainian
  - French-Ukrainian
  - Spanish-Ukrainian
  - Swedish-Ukrainian
    
### Named-entity recognition (NER)

* [MITIE NER Model](https://lang.org.ua/en/models/#anchor1)

### Word embeddings

* [Word2Vec](https://lang.org.ua/en/models/#anchor4)
* [GloVe](https://lang.org.ua/en/models/#anchor4)
* [LexVec](https://lang.org.ua/en/models/#anchor4)
