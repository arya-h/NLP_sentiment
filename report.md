# Intelligent Systems - *NLP deliverable*
### *Universidad Politecnica de Madrid - 2021-22*
#### Arya Houshmand <arya.houshmand@alumnos.upm.es>

#### Introduction
I have worked several times with formatted data (from sensors, ), but i always found human text too difficult to process and despite some courses I took during these past few years, the subject was stil very confusing.

As soon as I started following the NLP course I knew i wanted to build my own version of a sentiment analysis tool, which I had seen in blogs and other websites but never touched it with hand.

#### Dataset

- The dataset used is <a href ="https://www.kaggle.com/kritanjalijain/amazon-reviews?select=train.csv"> this one </a>. 
- It's already divided in train and test set, but the size of the files was too large for my computer's processing capabilities. I used *Google Colab* and *Kaggle* as 3rd party processing tools but even there I would have to wait for days on my free plan.
- The structure is fairly simple
  - **polarity** : corresponds to class index, already present in dataset. 1 for negative, 2 for positive
  - **title** : title of the review
  - **text** : body of the review


#### Framework and tools used
- I used python instead of R simply because I'm more accustomed to it and I already had knowledge with pandas.
- The libraries used were 
  - **spacy** : for tokenization, stopwords, basic ml models
  - **pandas** : to handle the large cvs
  - **nltk** : used for stopwords for the cloudword, but not used here for explicit tokenization. All the work was being done by *spacy*.

- I used Google Colab to handle the processing my computer wasn't able to suffice. I will link the notebook later on.


## Workflow

#### 1) <span style=color:#962121> File reading and processing </span>

- The file reading section is pretty straightforward. I'll start from the train set. The data was already clean so there wasn't a lot of processing required.

```python
def read_file(nrows):
    #read from csv, add column names
    df = panda.read_csv('colab_train.csv',
                         nrows=nrows, 
                         names=["polarity", "title", "text"])
    #the dataset provided was already clean, but i 
    #thought it was still good practice to clean up empty cells
    #remove rows with empty text
    df['text'].replace('', np.nan, inplace=True)
    df['title'].replace('', np.nan, inplace=True)
    #drop null values
    df.dropna(subset=['text', 'title'], inplace=True)

    #shuffle the whole dataframe to avoid having reviews only 
    #from a certain subset, since i will only take a subset of #these 2GBs
    df = df.sample(frac=1).reset_index(drop=True)

    return df
```

- The dataframe is then converted into a list of tuples ***<string, label>*** , where the string is the body of the review and the label is an object with the following format
  *cats* (categories) is a required keyword for spacy implementation.
  ```python
  label : 
    {'cats': 
        {'pos': True, 
         'neg': False}
    }
  ```


```python
def format_test_data(nrows):
    revs = []
    test_df = panda.read_csv("colab_test.csv", names=['polarity', 'title', 'text'], nrows=nrows)

    for index, row in test_df.iterrows():
        if (row['text']).strip():
            if row['polarity'] == 2:
                lab = "pos"
            else:
                lab = "neg"
            spacy_label = {
                'cats': {
                    "pos": lab == "pos",
                    "neg": lab == "neg"
                }
            }
            revs.append((row['text'], spacy_label))

    # shuffle test set
    random.shuffle(revs)

    return revs
```








