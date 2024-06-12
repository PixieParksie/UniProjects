2023 August

## Purpose of this project:
To compare the performance of three classification algorithms Multinomial Naive Bayes, Multilayer Perceptron, and Support Vector Machine 
<br>

## Tools
Python 

<br>

## Algorithms and parameters used:

- Multinomial Naive Bayes (MNB), Param: default
- Multilayer Perceptron (MLP), Param: 100,100, a=0.01
- Support Vector Machine (SVM), Param: default

<br>

## Results (accuracy):
- MNB: 0.533
- MLP: 0.53
- SVM: 0.59

<br>

## Best-performing algorithm and why:
Support Vector Machine.

Reason: SVM demonstrated the highest Accuracy, Precision and Recall rates. It also exhibited robustness to noise and changes in input data, making it the ideal choice for the this particular dataset. Additionaly, it does not require significant computational resources, making it an approachable option.

<br>

## Details about the dataset:
- This public domain dataset is collected from data.world platform, shared by @crowdflower, a data enrichment, mining and crowdsourcing company based in the US.
- The dataset consists of 40000 records of tweets labelled with 13 different sentiments, followed by Tweet ID number.
- All data types are strings, except for the 'tweet ID' column, which is an integer.
- There is a class imbalance of <21.32%. Tweets primarily convey neutral and negative sentiments.
- Contains 172 duplicate rows (tweets) but categorised with different sentiment labels.
- Word lengths vary, ranging from a minimum of 1 word to a maximum of 16 words.
- The data exhibits some degree of disorder and lack of cohesion. Various linguistic patterns used to express sentiment.
- Some information contains only special characters or hyperlinks.
- Contains informal and colloquial terms. For instance, “peeps” is a friendly term for “People”.
- Contains self-made terms, slangs or misspellings such as “Humpalow”.

<br>

## Four things I did to maximise accuracy and why:
- One: Cleaned special characters
 
Rationale: Tweets contain dynamic ways of conveying emotions through text. We retained punctuation marks ('!' and '?') to capture emotional tones during text cleaning. However, as there was no significant difference in the results, hence we proceeded to remove these characters.


- Two: Lemmatisation
  
Rationale: Experimented stemming and lemmatisation, using both and interchangeably. Neither had zero to minimal impact on accuracy. Stemming showed no effect, while lemmatisation produced slight changes of 0.001.


- Three: Retained stopword

Rationale: Retaining stopwords improved precision. There's a possibility that removing stopwords could result in the loss of valuable information, especially in rows where a high number of stopwords, accounting for over 50% of the sentence.


- Four: TF-IDF as Feature Engineering method

Rationale: TF-IDF exhibited higher accuracy compared to GloVe. Accuracy rates dropped by up to 20% after embedding text using GloVe.

<br>

## What did I do to mitigate data imbalance?:

I Combined the 13 labels into three primary emotions: Positive, Negative, and Neutral. Further details are explained in [here](#Data-Exploration-Balancing-Class).

<br>

## Table of Contents:

#### 1. [Data Exploration](#Data-Exploration) 
- Preparaion
- Balancing class

#### 2. Preprocess
- Data cleaning
- Exploring Cleaned Data & Investigating Stopwords in Text
- Tokenising

#### 3. Feature Engineering
- Vectorising using TF-IDF

#### 4. Model Training/Testing/Evaluation
- MNB
- MLP
- SVM


```python
!pip show nltk
!pip install --upgrade nltk
```

```python
import pandas as pd
import numpy as np

import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
!pip install stop_words
nltk.download('stopwords')
import pickle

from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
#!pip install stop_words

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt
```

```python
df = pd.read_csv('tweet_emotions.csv')
df.head(10)
```

## Data Exploration
```python
df.info()
```

```python
# Find any numerical variables.
numerical = [var for var in df.columns if df[var].dtype!='O']

print('Num of numerical variables: {}\n'.format(len(numerical)))
print('The numerical variables are: ', numerical)
```

```python
# Drop the unecessary column (numerical variables)
df = df.drop(columns=['tweet_id'])
```

```python
# Count duplicates
duplicates_count = df.duplicated().sum()
print(f'Total duplicated rows: {duplicates_count}')
```

```python
# Check for duplicates
df[df['content'].duplicated() == True]
```

```python
# Drop duplicated values
index = df[df['content'].duplicated() == True].index
df.drop(index, axis = 0, inplace = True)
df.reset_index(inplace=True, drop = True)
```

```python
# Final shape of data after dropping duplicates
df.shape
```

## Data Exploration - Exploring Class
```python
# Unique values from 'sentiment'
unique_sentiments = df['sentiment'].unique()
print(unique_sentiments)
```

```python
# Frequency distribution of'sentiment'
frequency_counts = df['sentiment'].value_counts()

frequency_percentage = (frequency_counts / len(df['sentiment'])) * 100
frequency_df = pd.DataFrame({'Counts': frequency_counts, 'Percentage': frequency_percentage})

print(frequency_df)

# Print total value
cardinality = df['sentiment'].nunique()
print(f"\ntotal values: {cardinality}")
```

```python
# Extract sentiment values & frequencies
sentiment_counts = df['sentiment'].value_counts().sort_index()

sentiments = sentiment_counts.index
frequencies = sentiment_counts.values

# Plot
plt.barh(sentiments, frequencies, color='orange')
plt.xlabel('Frequency')
plt.ylabel('Sentiment')
plt.title('Distribution of class "sentiment"')
plt.gca().invert_yaxis()  # Invert the y-axis to have the highest sentiment at the top
plt.show()
```
![Distribution class sentiment](https://github.com/PixieParksie/-Uni-Project-Data-Mining-/assets/106667881/ab1e9164-3da2-422f-b61f-6ad4002f0c7a)


Observations:
- Classes are imbalanced - each class is not evenly distributed. Imbalance rate <21.32% (anger - neutral)
- Tweets primarily convey neutral and negative sentiments

## [Data Exploration] Balancing Class

The above result illustrates class imbalance of <21.32%. To find the best ratio, I've experimented with aggregating classes in five different methods:

<strong> Method 1: </strong>  Categorising the data into two primary emotions: Positive, Negative ('Surprise' and 'Neutral' as Negative).
![class distribution 1](https://github.com/PixieParksie/-Uni-Project-Data-Mining-/assets/106667881/a8c43201-6388-4687-a52f-d635bd5d608e)

<strong> Method 2: </strong> Categorising the data into two primary emotions: Positive, Negative ('Surprise' and 'Neutral' as Positive).
![class distribution 2](https://github.com/PixieParksie/-Uni-Project-Data-Mining-/assets/106667881/12c12575-18b3-48b9-84f2-c268294a4c86)

<strong> Method 3: </strong> Categorising the data into two primary emotions: Positive, Negative ('Neutral' as Positive, 'Surprise' as Negative).
![class distribution 3](https://github.com/PixieParksie/-Uni-Project-Data-Mining-/assets/106667881/5b538466-2977-4c23-aac5-31f066c5d2c6)

<strong> Method 4: </strong> Categorising the data into two primary emotions: Positive, Negative ('Surprise' as Positive, 'Neutral' as Negative).
![class distribution 4](https://github.com/PixieParksie/-Uni-Project-Data-Mining-/assets/106667881/5ee3469f-17d6-4d16-987d-07bb4f932725)

<strong> Method 5: </strong> Categorising the data into three primary emotions: Positive, Negative, and Neutral
![class distribution 5](https://github.com/PixieParksie/-Uni-Project-Data-Mining-/assets/106667881/8a036043-ea2b-4c12-8df9-43d48851c097)


When 'Surprise' and 'Neutral' are classified under a separate class 'Neutral', we encounter an imbalance of up to 13.17%, slightly above the general guideline.

(General guideline: data is considered imbalanced when prevalence ≤ 10%)

Among the 5 results, method 3 yielded the lowest imbalance at 8.58%. However, I would not choose this method primarily because 'neutral' and 'surprise' rely significantly on the context of the text for differentiation, unlike the others, which are clearly either positive or negative.

'Neutral' typically represents a lack of strong emotion or a state of indifference. It's neither strongly positive nor negative.

'Surprise' denotes a sudden feeling of astonishment or unexpectedness. It's transient and could be either positive or negative, depending on the context in which it occurs. For instance, a surprise party might evoke positive emotions, while a surprising piece of news might evoke negative emotions.

Hence, it's not ideal to bias the neutral emotions towards either the positive or negative



```python
# Values from 'sentiment'
unique_sentiments = df['sentiment'].unique()
print(unique_sentiments)
```

```python
# Code for Method 5: grouping 13 labels into 3 primary classes: 'negative', 'positive', 'neutral'
sentiment_mapping = {
    'empty': 'negative',
    'sadness': 'negative',
    'worry': 'negative',
    'hate': 'negative',
    'boredom': 'negative',
    'anger': 'negative',
    'enthusiasm': 'positive',
    'neutral': 'neutral',
    'surprise': 'neutral',
    'love': 'positive',
    'fun': 'positive',
    'happiness': 'positive',
    'relief': 'positive'
}

# Create a new col for the mapping
df['label'] = df['sentiment'].map(sentiment_mapping)
```

```python
# Drop the original column, 'sentiment'
df = df.drop(columns=['sentiment'])
```

```python
# Final check - Extract unique values from 'sentiment'
import pandas as pd

unique_sentiments = df['label'].unique()
print(unique_sentiments)
```

```python
df.head(5)
```

```python
# Calculate the frequency distribution of 'sentiment'
frequency_counts = df['label'].value_counts()

frequency_percentage = (frequency_counts / len(df['label'])) * 100
frequency_df = pd.DataFrame({'Counts': frequency_counts, 'Percentage': frequency_percentage})

print(frequency_df)

# Total value count
cardinality = df['label'].nunique()
print(f"\ntotal values: {cardinality}")
```


```python
sentiment_counts = df['label'].value_counts().sort_index()

# Extract sentiment values & frequencies
sentiments = sentiment_counts.index
frequencies = sentiment_counts.values

total = sum(frequencies)  # Calculate the total frequency

plt.barh(sentiments, frequencies, color='orange')
plt.xlabel('Frequency')
plt.ylabel('Sentiment')
plt.title('Distribution of the combined classes (sentiment)')
plt.gca().invert_yaxis()

# Annotate each bar with its percentage
for index, value in enumerate(frequencies):
    percentage = (value / total) * 100
    plt.text(value - (max(frequencies) * 0.09), index, f'{percentage:.2f}%', ha='center', va='center', color='black')

plt.show()
```
![class distribution 6 - after combining](https://github.com/PixieParksie/-Uni-Project-Data-Mining-/assets/106667881/5cec39df-dac3-4a21-8bfd-e1a2b5b32500)


```python
# Mapping sentiment num / encode
df["label_num"] = df.label.map({
    'negative': 0,
    'positive': 1,
    'neutral': 2
})

df = df.drop(columns=['label'])
df.head(5)
```

```python
X = df['content']
y = df['label_num']

# Making sure that X y have the same length
print(len(X))
print(len(y))
```

## [Preprocessing] Data Cleaning
```python
# Cleaning and lemmatising
cleaned = []

for sen in range(0, len(X)):
    # Remove all the special characters (any letter or a digit)
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters (surrounded by whitespace)
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization- splits into list of words ['The', 'quick', ....]
    document = document.split()

    lemma = WordNetLemmatizer()
    document = [lemma.lemmatize(word) for word in document]
    document = ' '.join(document)
    cleaned.append(document)
```

```python
# Making sure that cleaned data has the same length as X
len(cleaned)
```

## Exploring Cleaned Data & Investigating Stopwords in Text
```python
# Count stopwords present in the data
nltk.download('punkt')
def count_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = nltk.word_tokenize(text)
    return len(set(words) & stop_words)

# Add a new col 'stop_words' that contains the count of stopwords
df['stop_words'] = df['content'].apply(count_stopwords)
value_counts = df['stop_words'].value_counts()
```

```python
# Count the number of stopwords in the data
temp = df.copy()
stop_words = set(stopwords.words("english"))
temp['stop_words'] = temp['content'].apply(lambda x: len(set(x.split()) & stop_words))

# Print rows that contain stopwords
rows_with_stopwords = temp[temp['stop_words'] > 0]
print(rows_with_stopwords)
```

```python
# Explore tweet that contains stopwords
df['content'][3]
# df.loc[3]
```

```python
# Count stopwords in index 3
stop_words = set(stopwords.words("english"))
df_indx = df['content'][3]
stopword_count = len([word for word in df_indx.split() if word in stop_words])
stopword_count
```

```python
# Print all the stopwords in index 3
stopwords_in_content = [word for word in df_indx.split() if word in stop_words]

print("Stopwords in df['content'][3]:")
print(stopwords_in_content)

'''
The cleaned sentence from 4th row, ['wants to hang out with friends SOON!']
contains 3 stopwords, 'to', 'out', and 'with'
'''
```

## Preprocessing: Tokenisation
```python
cleaned_tokenized = []
for each in cleaned:
    doc = sent_tokenize(each)
    for sentence in doc:
        cleaned_tokenized.append(sentence)

# Type and length of the cleaned & tokenised sentence
print(type(cleaned_tokenized))
print(len(cleaned_tokenized))

# Print the fist 15 lines of the cleaned & tokenised sentence
for i in range(15):
  print(cleaned_tokenized[i])
```

```python
# Crate a new colum that contains processed text
df['processed_content'] = cleaned_tokenized

# Reorder col
desired_order = ['content', 'processed_content', 'label_num']
df = df[desired_order]
df.head(10)

X = cleaned_tokenized
```

```python
# Pick a random row, count SW in that sentence
df.loc[3]
```

## Feature Engineering: Vectorising Using TF-IDF
```python
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X)
freq_term_matrix = count_vectorizer.transform(X)      # CV sparse matrix

tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix) # tf-idf sparse matrix
# print(tf_idf_matrix)
dense_tf_idf_matrix = tf_idf_matrix.toarray()         # tf-idf dense matrix
# print(dense_tf_idf_matrix)

X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y, test_size=0.2, random_state=42)
```

## Model Training/Testing/Evaluation: MNB
```python
model = MultinomialNB()

# Train
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

# Evaluate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)

classification_rep = classification_report(y_train, y_pred_cv)
print("Classification Report (Cross-Validation):\n", classification_rep)

accuracy = accuracy_score(y_train, y_pred_cv)
print("Accuracy (Cross-Validation):", accuracy)

model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)

classification_rep_test = classification_report(y_test, y_pred_test)
print("Classification Report (Test Data):\n", classification_rep_test)

accuracy_test = accuracy_score(y_test, y_pred_test)
print("Accuracy (Test Data):", accuracy_test)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculating the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix NB')
plt.show()
```

## Model Training/Testing/Evaluation: MLP
```python
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 100), alpha = 0.01, max_iter=100)

# Train
history = mlp_model.fit(X_train, y_train)

# Test
y_pred = mlp_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

train_loss = history.loss_curve_
num_epochs = len(train_loss)

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

## Model Training/Testing/Evaluation: SVM
```python
# Train
model = SVC()
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

# Evaluate
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```


