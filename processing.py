import random

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as panda
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import spacy
from spacy.util import minibatch, compounding

def read_file(nrows):
    #read the csv file, for development will keep nrows = low
    #csv format is
    #polarity : 1--> negative // 2--> positive
    #title
    #text
    df = panda.read_csv('data/train.csv', nrows=nrows, names=["polarity", "title", "text"])
    #remove rows with empty text
    df['text'].replace('', np.nan, inplace=True)
    df['title'].replace('', np.nan, inplace=True)
    #drop null values
    df.dropna(subset=['text', 'title'], inplace=True)
    #shuffle it
    df = df.sample(frac=1).reset_index(drop=True)

    return df

#not used, kept for future reference
def processing(df):
    #word tokenization
    token_list = []
    token_list_filt = []
    nlp = spacy.load("en_core_web_sm")
    for index, row in df.iterrows():
        doc = nlp(row['text'])
        #token_list = [token for token in doc]
        # remove stopwords with spacy
        token_list_filt = [token for token in doc if not token.is_stop]

    #normalization, lemmatization with spacy

    lemmas = [f"token :{token}, lemma : {token.lemma_}" for token in token_list_filt]
    return lemmas

def wordcloud(df):
    #separate negative and positive dfs
    # stopwords = set(STOPWORDS)
    # stopwords.update(['book', 'read', 'good', 'movie', 'dvd', 'make'
    #                   , 'great', 'time'])
    # negative = df[df.polarity == 1]
    # positive = df[df.polarity == 2]
    #
    # textPos = " ".join(rev for rev in positive.text)
    # wcPos = WordCloud(stopwords=stopwords).generate(textPos)

    #textNeg = " ".join(rev for rev in negative.text)
    #wcNeg = WordCloud(stopwords=stopwords).generate(textNeg)
    #
    # plt.imshow(wcPos, interpolation='bilinear')
    # plt.show()

    # df['polarityt'] = df['polarity'].replace({1: 'negative'})
    # df['polarityt'] = df['polarityt'].replace({2: 'positive'})
    # fig = px.histogram(df, x="polarityt")
    # fig.update_traces(marker_color="indianred", marker_line_color='rgb(8,48,107)',
    #                   marker_line_width=1.5)
    # fig.update_layout(title_text='Product polarity')
    # fig.show()
    pass

def labeling(df):
    revs = []
    for index, row in df.iterrows():
        if(row['text']).strip():
            #label dictionary needed by spacy
            if row['polarity'] == 2:
                lab = "pos"
            else:
                lab = "neg"
            spacy_label = {
                'cats':{
                    "pos": lab == "pos",
                    "neg": lab == "neg"
                }
            }
            revs.append((row['text'], spacy_label))
            #print("text : {} \nlabel :{}".format(row['text'], spacy_label))

    #shuffle
    random.shuffle(revs)

    return revs

def training(data: list, iterations : int, test_data : list) -> None:

    #pipeline for spacy
    #useses a convolutional NN
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture" : "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    #train the category classifier
    train_no_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(train_no_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
        for i in range(iterations):
            print("iteration {}".format(i))
            loss = {}
            random.shuffle(data)
            batches = minibatch(data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(
                    text,
                    labels,
                    drop=0.2,
                    sgd=optimizer,
                    losses=loss
                )
            print("now evaluating for iteration {}".format(i))
            with textcat.model.use_params(optimizer.averages):
                #evaluate the updated model
                evaluation_res = evaluate_model(tokenizer=nlp.tokenizer,
                                               textcat = textcat,
                                               test_data=test_data)

                #for every iteration print out a grid of the loss / accuracy / recall / f-score
                print(f"{'Loss: ' + str(loss['textcat']):<34} Accuracy: { str(evaluation_res['precision'])}")
                print(f"{'Recall : '+ str(evaluation_res['recall']):<34} F-score: {str(evaluation_res['f-score'])}")


    #save model
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")


def evaluate_model(
    tokenizer, textcat, test_data: list
) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # not 0 because it will be present in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]
        true_label = true_label.get('cats')
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if (
                predicted_label == "neg"
            ):
                continue
            if score >= 0.5 and true_label["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


def format_test_data(nrows):
    revs = []
    test_df = panda.read_csv("data/test.csv", names=['polarity', 'title', 'text'], nrows=nrows)

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

    # shuffle
    random.shuffle(revs)

    return revs

def test_model(input_data: str):
    #  Load saved trained model
    loaded_model = spacy.load("model_500kk/model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )
    return (prediction, score)


if __name__=="__main__":
    df = read_file(10)
    print("File read")
    #lemmas = processing(df)
    labeled_revs = labeling(df)
    # for text, rat  in labeled_revs:
    #     print("text : {}\nlabel : {}".format(text, rat))
    print("labeling done")
    list_test_data = format_test_data(10)
    print("read test data")
    training(labeled_revs, 20, list_test_data)

    test_string = "All the wipes are mostly dry. its like mostly paper towels. Instead of buying this we " \
                  "can use paper napkins. HUGGIES from Costco store were out of stock in my area." \
                  " So i ended up buying here. we have to use it for furniture, car cleaning etc now."

    test_string2 = "My six-month-old granddaughter loves this toy. She’s too young for stacking, " \
                   "but enjoys exploring the different rings. They have different colors and textures, " \
                   "and her favorite is the clear rattle. Very thoughtfully designed toy that will provide" \
                   " several years of fun and learning."

    test_string3 = "I REALLY like this! "

    test_model(test_string)




