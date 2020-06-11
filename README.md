
# Web Application For Sentiment Analysis Of Tweets

## Project Overview
This is a application for sentiment analysis of tweets and creating maps and histograms based on it.
---
\
![Number Of Tweets](https://github.com/pranaykankariya97/Web-App-For-Sentiment-Analysis-Of-Tweets/blob/master/Number%20of%20Tweets.png)
\
---
![Location Of Tweets](https://github.com/pranaykankariya97/Web-App-For-Sentiment-Analysis-Of-Tweets/blob/master/Map.png)
\
---
![Comparison between different Airlines](https://github.com/pranaykankariya97/Web-App-For-Sentiment-Analysis-Of-Tweets/blob/master/Histogram.png)


## Model for Sentiment Analysis
Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length.
In order to address this, scikit-learn provides utilities for the most common ways to extract numerical features from text content, namely:
* **tokenizing** strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators.
* **counting** the occurrences of tokens in each document.
* **normalizing** and weighting with diminishing importance tokens that occur in the majority of samples / documents.
A corpus of documents can thus be represented by a matrix with one row per document and one column per token (e.g. word) occurring in the corpus.
We call **vectorization** the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the **Bag of Words** or **Bag of n-grams** representation.\

`tf(t,d) - no of times a term 't' occurs in a document 'd'`\
`idf(t,d) - inverse document frequncy`\
`tf-idf(t,d) = tf(t,d) x idf(t,d)`

## How To Run This Application
After downloading the file, type in your Command prompt:\
`streamlit run US_Airlines_Tweets.py`

## Software And Libraries
This project uses the following software and libraries:
* [python 3.8.0](https://www.python.org/downloads/release/python-380/)
* [Jupyter Notebook](https://jupyter.org/)
* [Streamlit](https://www.streamlit.io/)
* [pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [pickle](https://docs.python.org/3/library/pickle.html)
* [Natural Language Toolkit](https://www.nltk.org/)

## Contact
Email: pranaykankariya97@gmail.com \
Project Link: [https://github.com/pranaykankariya97/Web-App-For-Sentiment-Analysis-Of-Tweets](https://github.com/pranaykankariya97/Web-App-For-Sentiment-Analysis-Of-Tweets)

