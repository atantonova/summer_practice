## Text message classifier

The problem is: 
Build a classifier of text messages so that it gives probability of message's subject being '*finance*'.
Then for messages with high probability model gives sentiment: *positive/neutral/negative*.

#### Folders
**code** - jupyter notebooks for data cleaning and model learning and evaluation

**data** - datasets, original and edited

[Link to Colab Notebook with learning and evaluation](https://colab.research.google.com/drive/1JAieTnBlVFxX7bnOXiW04GYMJxQhs0xD?usp=sharing)

[Link to notebook with prediction on trained model](https://colab.research.google.com/drive/1e9ReN9jksHcrRTJJiCJOahHPynkFoRlJ?usp=sharing)

---
## Usage

[predict.ipynb](./code/predict.ipynb) contains functions for 
- data cleaning
- loading tokenizers and models
- predicting classification and sentiment

```python
classifier, tokenizer_classifier, model_sent, tokenizer_sent = load_model_tokenizer(path_to_classifier, path_to_tokenizer)
test_data = pd.read_csv(path_to_test_data)
test_data = clean_data(test_data)
results = predict_with_sentiment(test_data, classifier, tokenizer_classifier, model_sent, tokenizer_sent)
```
***path_to_classifier***: str - fine-tuned model can be found [here on Google Drive](https://drive.google.com/drive/folders/12KY0j5BUxfHuDsPxO68p46fwR_LXjdGU?usp=sharing)
***path_to_tokenizer***: str - tokenizer used in training and evaluation of classifier - *'bert-base-uncased'*
***test_data***: DataFrame - columns: 'text'
***results***: DataFrame - columns: 'text', 'finance_proba', 'positive', 'neutral', 'negative'


---

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

---

## Results

#### News Category dataset

Fine-tuned pretrained [BERT model for text classification](https://huggingface.co/transformers/model_doc/bert.html?highlight=bertforsequenceclassification#transformers.BertForSequenceClassification) with training paramenters from [this article](https://www.thepythoncode.com/article/finetuning-bert-using-huggingface-transformers-python) and [this doc](https://huggingface.co/transformers/custom_datasets.html) on balanced dataset I got this results on test data: 
```python
              precision    recall  f1-score   support

           0       0.80      0.78      0.79      1328
           1       0.79      0.81      0.80      1373

    accuracy                           0.79      2701
   macro avg       0.79      0.79      0.79      2701
weighted avg       0.79      0.79      0.79      2701
```

With imbalabnced dataset loss function was not decreasing much, accuracy was not growing for 5000 steps, so training was stopped.

#### Keyword extraction

I tried KeyBERT for keyword extraction from all-financial tweets and news dataset, found some bugs in data preprocessing, playing with paramenters of keywords search. 

Right now 40 most relevant keywords are contain 'bitcoin', 'insiders', 'trader', 'crypto' and other  connected to finance keywords and some strange ones: 'bénéteau', 'equitiesinc', 'itau'.

Next step is find out if these strange ones are company names, currency names (which is okay) or they are nicknames, websites (I cleaned data from them, but...) 

#### Combine financial tweets with no-topic in dataset

I had an assumption that in datasets with no-topic tweets there are not many financial tweets. So I combined 
- [Sentiment Analysis for Financial News](https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news) and [Twitter and Reddit Sentimental analysis Dataset](https://www.kaggle.com/cosmos98/twitter-and-reddit-sentimental-analysis-dataset) (Twitter part of it) into a trainig dataset - [dataset_for_training](./data/prepared/dataset_for_training.csv)
- [Sentiment Analysis on Financial Tweets](https://www.kaggle.com/vivekrathi055/sentiment-analysis-on-financial-tweets) and [Twitter and Reddit Sentimental analysis Dataset](https://www.kaggle.com/cosmos98/twitter-and-reddit-sentimental-analysis-dataset) (Reddit part of it) into a test dataset [dataset_for_testing](./data/prepared/dataset_for_testing.csv)
- [Sentiment Analysis on Financial Tweets](https://www.kaggle.com/vivekrathi055/sentiment-analysis-on-financial-tweets) and [News category dataset](https://www.kaggle.com/rmisra/news-category-dataset) with topic not business or money - [dataset_for_testing_v2](./data/prepared/dataset_for_testing_v2.csv)

Cleaned them from links, account names, numbers, punctuation, made datasets balanced, fine-tuned basic BERT classifier ('bert-base-uncased') on train, predicted on model after 700 steps of optimizer with results. 
This is reuslt on unseen data, 1000 samples. Result can be better after more steps, model is training at the moment. 

```python
              precision    recall  f1-score   support

           0       0.96      0.81      0.88       490
           1       0.84      0.97      0.90       510

    accuracy                           0.89      1000
   macro avg       0.90      0.89      0.89      1000
weighted avg       0.90      0.89      0.89      1000
```

This is result on second test dataset with 100 samples.

```python
              precision    recall  f1-score   support

           0       0.89      0.31      0.46        51
           1       0.57      0.96      0.72        49

    accuracy                           0.63       100
   macro avg       0.73      0.64      0.59       100
weighted avg       0.73      0.63      0.59       100
```

Result on hand-written test data

"sell ​​tesla it will fall soon",1
"you need to buy tesla shares i advise everyone",1
"bought a new scarf today",0
"will it rain tomorrow",0

is [1, 1, 1, 0].

---
## Sentiment analysis

Second part of the problem was to implement sentiment analysis for messages with high 'finance' probability. 
For this I used pretrained model [FinBERT](https://github.com/ProsusAI/finBERT). 