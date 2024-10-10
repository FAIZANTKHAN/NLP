# Natural Language Processing (NLP) Programs Repository

Welcome to the **NLP Programs Repository**, a comprehensive collection of code snippets and exercises covering essential Natural Language Processing (NLP) topics. This repository aims to help beginners and advanced users understand the core concepts of NLP and implement them using popular Python libraries such as **spaCy**, **NLTK**, **Gensim**, and others.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Topics Covered](#topics-covered)
    - [Regular Expressions (Regex)](#1-regular-expressions-regex)
    - [spaCy Overview](#2-spacy-overview)
    - [NLTK Overview](#3-nltk-overview)
    - [Language Processing Pipeline](#4-language-processing-pipeline)
    - [Stemming and Lemmatization](#5-stemming-and-lemmatization)
    - [Part-of-Speech (POS) Tagging](#6-pos-tagging)
    - [Named Entity Recognition (NER)](#7-named-entity-recognition-ner)
    - [Bag of Words (BOW)](#8-bag-of-words-bow)
    - [Stop Words](#9-stop-words)
    - [Bag of N-Grams](#10-bag-of-n-grams)
    - [TF-IDF](#11-tf-idf)
    - [Word Vectors Using spaCy](#12-word-vectors-using-spacy)
    - [Word Vectors Using Gensim](#13-word-vectors-using-gensim)
    - [Text Classification with Gensim](#14-text-classification-with-gensim)
3. [Exercises and Datasets](#exercises-and-datasets)
4. [Installation and Dependencies](#installation-and-dependencies)
5. [Contributing](#contributing)
6. [License](#license)

---

## Getting Started

To use the code and exercises in this repository, you need to set up a Python environment with all the required libraries. Ensure you have Python 3.x installed and follow the instructions below to install the dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/nlp-programs.git

# Navigate to the repository folder
cd nlp-programs

# Install the required libraries
pip install -r requirements.txt
```

The `requirements.txt` file contains all the necessary packages, such as **spaCy**, **NLTK**, **Gensim**, and others.

---

## Topics Covered

### 1. Regular Expressions (Regex)
Regex is a powerful tool for text manipulation, pattern matching, and preprocessing. This section covers how to use Python’s built-in `re` library to:
- Search for patterns
- Match and replace text
- Split text based on patterns

**Import:**
```python
import re
```

**Example:**
```python
# Find all words starting with 'p'
text = "Python programming is powerful"
matches = re.findall(r'\bp\w+', text)
print(matches)
```

### 2. spaCy Overview
**spaCy** is a powerful NLP library for advanced natural language processing tasks such as tokenization, POS tagging, NER, and more. This section introduces basic spaCy concepts and showcases how to use it for various NLP tasks.

**Install:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

**Import:**
```python
import spacy
nlp = spacy.load('en_core_web_sm')
```

### 3. NLTK Overview
The **Natural Language Toolkit (NLTK)** is one of the most popular libraries for text processing. This section will introduce basic NLTK concepts, such as tokenization, stop words, and frequency distribution.

**Install:**
```bash
pip install nltk
```

**Import:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 4. Language Processing Pipeline
This topic covers how to construct a full NLP pipeline for preprocessing text, using libraries like spaCy and NLTK to process raw text into structured data.

---

### 5. Stemming and Lemmatization
- **Stemming** refers to reducing a word to its root form.
- **Lemmatization** refers to reducing a word to its base or dictionary form.

This section shows how to perform stemming using **NLTK's PorterStemmer** and **spaCy's lemmatizer**.

**Example (PorterStemmer):**
```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_word = stemmer.stem("running")
```

**Example (spaCy Lemmatizer):**
```python
doc = nlp("running ran runner")
for token in doc:
    print(token.text, token.lemma_)
```

---

### 6. Part-of-Speech (POS) Tagging
POS tagging is the process of identifying the grammatical category of words. Learn how to perform POS tagging using **spaCy** and **NLTK**.

**Example (spaCy):**
```python
doc = nlp("The quick brown fox jumps over the lazy dog.")
for token in doc:
    print(token.text, token.pos_)
```

---

### 7. Named Entity Recognition (NER)
NER is a technique to identify named entities such as persons, organizations, and locations in text. This section demonstrates how to extract entities using **spaCy**.

**Example:**
```python
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

---

### 8. Bag of Words (BOW)
Bag of Words is a simple and widely used method for text representation. This section explains how to implement BOW using **scikit-learn**.

**Install:**
```bash
pip install scikit-learn
```

**Example:**
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(["This is a sentence", "This is another sentence"])
print(X.toarray())
```

---

### 9. Stop Words
Stop words are commonly used words (like "and", "the") that are often removed from text data before processing. This section covers stop word removal using **spaCy** and **NLTK**.

**Example:**
```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
```

---

### 10. Bag of N-Grams
N-grams are sequences of tokens of length N. This section explains how to create n-grams for text analysis.

---

### 11. TF-IDF
TF-IDF is a statistic that reflects the importance of a word in a document relative to a corpus. This section covers how to compute TF-IDF using **scikit-learn**.

---

### 12. Word Vectors Using spaCy
This section demonstrates how to access word vectors in **spaCy** to find word similarities.

---

### 13. Word Vectors Using Gensim
**Gensim** is a popular library for word embeddings. This section shows how to generate and use word vectors with Gensim's Word2Vec.

**Install:**
```bash
pip install gensim
```

---

### 14. Text Classification with Gensim
This section covers a simple text classification task using **Gensim** and **Word2Vec** embeddings.

---

## Exercises and Datasets
This repository includes various exercises related to each topic. You can practice your skills using datasets from:
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle](https://www.kaggle.com/)

---

## Installation and Dependencies
To install all the dependencies for the repository:
```bash
pip install -r requirements.txt
```

---

## Contributing
Contributions are welcome! Please feel free to submit a pull request.

---

## License
This repository is licensed under the MIT License.
