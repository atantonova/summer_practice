## Text message classifier

The problem is: 
Build a classifier of text messages so that it gives probability of message's subject being '*finance*'.
Then for messages with high probability model gives sentiment: *positive/neutral/negative*.

#### Folders
**code** - jupyter notebooks for data cleaning and model learning and evaluation

**data** - datasets, original and edited

**notes** - notes on article review, searching of datasets and thier cleaning, model learning and evaluation

[Active link to Colab Notebook with learning and evaluation](https://colab.research.google.com/drive/1JAieTnBlVFxX7bnOXiW04GYMJxQhs0xD?usp=sharing)

### Datasets: find and prepare for learning
#### Finding ready-to-go dataset

The most suitable dataset I found was [News category dataset](https://www.kaggle.com/rmisra/news-category-dataset). 
I decided that categories 'BUSINESS' and 'MONEY' are close to 'finance' and made 2 datasets:
- imbalanced with all data [here](./data/news_cleaned.csv), categories 'BUSINESS' and 'MONEY' are 1 in 'financial' column, other - 0 
- balanced with proportionally (by other categories) selected data, same label encoding [here](./data/news_cleaned_balanced.csv)

#### Creating new dataset from "non-suitable"

Sicne there is no perfect dataset I tried to create dataset for classification from datasets for sentiment analysis on tweets/news/comments

1. Financial tweets/news datasets
   These datasets are focused on financial tweets sentiment analysis
   [Sentiment Analysis for Financial News](https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news)
   [Sentiment Analysis on Financial Tweets](https://www.kaggle.com/vivekrathi055/sentiment-analysis-on-financial-tweets)

2. Tweets/news/comment for sentiment analysis without particular topic
   [Twitter and Reddit Sentimental analysis Dataset](https://www.kaggle.com/cosmos98/twitter-and-reddit-sentimental-analysis-dataset)
   [Enron Corpus database for E-mail classification](https://github.com/anthdm/ml-email-clustering/blob/master/split_emails.csv)

All of these datasets are for sentimental analysis because this kind of problem is the most common for tweets/news/comments analysis

***Idea 1***
1. Take dataset with tweets on financial topic, extract keywords
2. Make new dataset from datasets without topic and finance datasets
3. Run BERT text classification for fine tunung

***Idea 2***
1. Make new dataset from datasets without topic and finance datasets
2. Perform unsupervised cluster analysis
3. Find cluster with most relevant data for finance, mark it as finance, other non-finance
4. Run BERT text classification for fine tunung

