# Financial Sentiment Analysis Project

This project aims to analyze the sentiment of financial news articles using natural language processing and machine learning techniques. We will be exploring different methods of representing the text data, including TF-IDF, pre-trained Google News Word2Vec features, and Doc2Vec, and comparing the performance of various classification models, including SVM, KNN, and Random Forest, with cross-validation to tune hyperparameters.

## Data

The dataset used in this project consists of financial news articles from a variety of sources. The data is open source and can be found https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news. 

## Data Preprocessing

Before we can begin training our models, we need to preprocess the data. This includes tasks such as removing stop words, stemming, and converting the text data into numerical vectors.

## Feature Representation

We will be exploring three different methods of representing the text data:

1. TF-IDF: We will be using TF-IDF to convert the text data into numerical vectors. We will visualize the data using PCA and t-SNE to see how well the vectors separate the different sentiment classes.

2. Pre-trained Google News Word2Vec: We will be using pre-trained Google News Word2Vec features to represent the documents. We will compare the performance of using the max and average of the word vectors to represent each document.

3. Doc2Vec: We will also explore using Doc2Vec to represent the documents.

## Classification Models

We will be using SVM, KNN, and Random Forest classification models to predict the sentiment of the financial news articles. We will use cross-validation to tune the hyperparameters of each model.

## Results

We will display the results of each model in a table, showing the accuracy, precision, recall, and F1-score for each class.

## Conclusion

In this project, we explored different methods of representing text data and compared the performance of different classification models for financial sentiment analysis. We found that TF-IDF with randomforest gives best results, followed by Max word2vec, doc2vec and average word2vec

## Future work

Concerning the future scope of the project, an argument can be made that these results can be improved using artificial neural networks such as Convolution neural networks(CNNs) and Transformers, as CNNs can capture local semantics and relationships between consecutive words and Transformers use a self-attention system to capture the relationships between all input tokens at the same time, allowing them to model long-term dependencies and effectively collect contextual information.
