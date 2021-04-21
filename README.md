# awesome-ukrainian-nlp
Curated list of Ukrainian natural language processing (NLP) resources (corpora, pretrained models, libriaries, etc.)


## 1. Datasets / Corpora

### Monolingual

* [Brown-UK](https://github.com/brown-uk/corpus) — carefully curated corpus of modern Ukrainian language
* [UberText](https://lang.org.ua/uk/corpora/#anchor4) — 6 GB of news, Wikipedia and fiction texts
* [Wikipedia](https://dumps.wikimedia.org/ukwiki/latest/)
* [OSCAR](https://oscar-corpus.com/) — shuffled sentences extracted from [Common Crawl](https://commoncrawl.org/) and classified with a language detection model. Ukrainian portion of it is 28GB deduplicated.
* [CC-100](http://data.statmt.org/cc-100/)  — documents extracted from [Common Crawl](https://commoncrawl.org/), automatically classified and filtered. Ukrainian part is 200M sentences or 10GB of deduplicated text.

### Labeled

* [UA-GEC](https://github.com/grammarly/ua-gec) —  grammatical error correction (GEC) and fluency corpus.
* [NER-uk](https://github.com/lang-uk/ner-uk) — Brown-UK labeled for named entities
* [Yakaboo Book Reviews](https://yakaboo-book-reviews-dataset.imfast.io/) — book reviews, rating and descriptions
* [Universal Dependencies](https://github.com/UniversalDependencies/UD_Ukrainian-IU/tree/master) — dependency trees corpus
 
### Dictionaries

* [ВЕСУМ](https://github.com/brown-uk/dict_uk) — POS tag dictionary. Can generate a list of all wordforms valid for spelling.
* [Tonal dictionary](https://github.com/lang-uk/tone-dict-uk)




## 2. Tools

* [tree_stem](https://github.com/amakukha/stemmers_ukrainian) — stemmer
* [pymorphy2](https://github.com/kmike/pymorphy2) + [pymorphy2-dicts-uk](https://pypi.org/project/pymorphy2-dicts-uk/) — POS tagger and lemmatizer
* [LanguageTool](https://languagetool.org/uk/) — grammar, style and spell checker
* [Stanza](https://stanfordnlp.github.io/stanza/) — POS, tokenization, lemmatization, etc. 

 

## 3. Pretrained models

### Language models

* [RoBERTa](https://huggingface.co/youscan/ukr-roberta-base)

### Machine translation

* [Helsinki NLP models](https://huggingface.co/Helsinki-NLP) — 10 language pairs, Ukrainian from/to English, Finnish, French, Spanish, Swedish.
* [M2M-100](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100) — translate from/to any of 100 languages.

### Sequence-to-sequence models

* [mBART50](https://github.com/pytorch/fairseq/tree/master/examples/multilingual#mbart50-models)
* [mT5](https://github.com/google-research/multilingual-t5)

### Named-entity recognition (NER)

* [MITIE NER Model](https://lang.org.ua/en/models/#anchor1)

### Word embeddings

* [fastText](https://fasttext.cc/docs/en/crawl-vectors.html)
* [fastText_multilingual](https://github.com/babylonhealth/fastText_multilingual) — word vectors in 78 languages, aligned to the same vector space.
* [Word2Vec](https://lang.org.ua/en/models/#anchor4)
* [GloVe](https://lang.org.ua/en/models/#anchor4)
* [LexVec](https://lang.org.ua/en/models/#anchor4)
