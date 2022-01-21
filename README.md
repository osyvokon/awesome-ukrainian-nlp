# awesome-ukrainian-nlp
Curated list of Ukrainian natural language processing (NLP) resources (corpora, pretrained models, libriaries, etc.)


## 1. Datasets / Corpora

### Monolingual

* [Brown-UK](https://github.com/brown-uk/corpus) — carefully curated corpus of modern Ukrainian language
* [UberText](https://lang.org.ua/uk/corpora/#anchor4) — 6 GB of news, Wikipedia and fiction texts
* [Wikipedia](https://dumps.wikimedia.org/ukwiki/latest/)
* [OSCAR](https://oscar-corpus.com/) — shuffled sentences extracted from [Common Crawl](https://commoncrawl.org/) and classified with a language detection model. Ukrainian portion of it is 28GB deduplicated.
* [CC-100](http://data.statmt.org/cc-100/)  — documents extracted from [Common Crawl](https://commoncrawl.org/), automatically classified and filtered. Ukrainian part is 200M sentences or 10GB of deduplicated text.
* [Ukrainian Twitter corpus](https://github.com/saganoren/ukr-twi-corpus) - Ukrainian Twitter corpus for toxic text detection.
* [Ukrainian forums](https://github.com/khrystyna-skopyk/ukr_spell_check/blob/master/data/scraped.txt) — 250k sentences scraped from forums.
* [Ukrainian Descriptions Of Words](https://www.kaggle.com/ihelon/ukrainian-descriptions-of-words) - kaggle-based small dataset (119 pairs) of Ukrainian word descriptions (like the game Alias). Can be used for semantics-related tasks.
* [Ukrainian news articles](https://www.kaggle.com/c/ukrainian-news-classification/overview) - 50K news articles with titles and sources (benchmark for text classification)

### Parallel

* [Polish-Ukrainian Parallel Corpus](https://clarin-pl.eu/dspace/handle/11321/535) - 1,156,579 words.
* [English-Ukrainian Tatoeba sentence pairs](https://www.manythings.org/anki/) - a collection of 155K sentence pairs of small length from a flashcards game.
* Some multilingual corpora are available on https://opus.nlpl.eu/, although many datasets are frequently misaligned.
* Ukrainian-English eval set of 997 sentences from [2021 shared task on multilingual translation](https://www.statmt.org/wmt21/large-scale-multilingual-translation-task.html).

### Labeled

* [UA-GEC](https://github.com/grammarly/ua-gec) — grammatical error correction (GEC) and fluency corpus.
* [NER-uk](https://github.com/lang-uk/ner-uk) — Brown-UK labeled for named entities
* [Yakaboo Book Reviews](https://yakaboo-book-reviews-dataset.imfast.io/) — book reviews, rating and descriptions
* [Universal Dependencies](https://github.com/UniversalDependencies/UD_Ukrainian-IU/tree/master) — dependency trees corpus
 
### Dictionaries

* [ВЕСУМ](https://github.com/brown-uk/dict_uk) — POS tag dictionary. Can generate a list of all word forms valid for spelling.
* [Tonal dictionary](https://github.com/lang-uk/tone-dict-uk)
* [Multilingualsentiment, includes Ukrainian](https://sites.google.com/site/datascienceslab/projects/multilingualsentiment) - a list of positive/negative words
* [obscene-ukr](https://github.com/saganoren/obscene-ukr) — profanity dictionary
* [Word stress dictionary](https://github.com/lang-uk/ukrainian-word-stress-dictionary) — word stress for 2.7M word forms 
* [Heteronyms](https://github.com/lang-uk/ukrainian-heteronyms-dictionary) — words that share the same spelling but have different meaning/pronunciation.
* [Abbreviations](https://github.com/asivokon/ukrainian-abbreviations-dictionary) — map abbreviation to expansion


## 2. Tools

* [tree_stem](https://github.com/amakukha/stemmers_ukrainian) — stemmer
* [pymorphy2](https://github.com/kmike/pymorphy2) + [pymorphy2-dicts-uk](https://pypi.org/project/pymorphy2-dicts-uk/) — POS tagger and lemmatizer
* [LanguageTool](https://languagetool.org/uk/) — grammar, style and spell checker
* [Stanza](https://stanfordnlp.github.io/stanza/) — Python package for tokenization, multi-word-tokenization, lemmatization, POS, dependency parsing, NER

 

## 3. Pretrained models

### Language models

* [RoBERTa](https://huggingface.co/youscan/ukr-roberta-base)
* [GPT-2](https://huggingface.co/Tereveni-AI/gpt2-124M-uk-fiction)

### Machine translation

* [Helsinki NLP models](https://huggingface.co/Helsinki-NLP) — 10 language pairs, Ukrainian from/to English, Finnish, French, Spanish, Swedish.
* [M2M-100](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100) — translate from/to any of 100 languages.

### Sequence-to-sequence models

* [mBART50](https://github.com/pytorch/fairseq/tree/master/examples/multilingual#mbart50-models)
* [mT5](https://github.com/google-research/multilingual-t5)

### Named-entity recognition (NER)

* [MITIE NER Model](https://lang.org.ua/en/models/#anchor1)

### Word embeddings

* [UA fastText on CommonCrawl and Wiki](https://fasttext.cc/docs/en/crawl-vectors.html)
* [UA fastText on Wiki](https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md)
* [fastText_multilingual](https://github.com/babylonhealth/fastText_multilingual) — word vectors in 78 languages, aligned to the same vector space.
* [Word2Vec](https://lang.org.ua/en/models/#anchor4)
* [GloVe](https://lang.org.ua/en/models/#anchor4)
* [LexVec](https://lang.org.ua/en/models/#anchor4)
* [BPEmb: Subword Embeddings, includes Ukrainian](https://nlp.h-its.org/bpemb/) - easy to use with [Flair](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/BYTE_PAIR_EMBEDDINGS.md)


## 4. Paid

* [LORELEI Ukrainian Representative Language Pack](https://catalog.ldc.upenn.edu/LDC2020T24) - Ukrainian monolingual text, Ukrainian-English parallel text, partially annotated for named entities
