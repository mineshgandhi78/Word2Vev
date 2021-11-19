from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re
from pandas import read_csv
from pandas import DataFrame as pd

stopwords_list = stopwords.words("english")

def change_lower(text):
    text = text.lower()
    return text

def clean_data(text):
    # text = re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', text)
    # text = re.sub(r'[\\/×\^\]\[÷]', '', text)
    text = " ".join(re.split(r"\W+", str(text).lower().strip()))

    return text

def train_w2v(w2v_df):
    w2v_model = Word2Vec(min_count=4)
    w2v_model.build_vocab(w2v_df, progress_per=10000)
    w2v_model.train(w2v_df, total_examples=w2v_model.corpus_count, epochs=1, report_delay=1)
    return w2v_model

def word2vec(df):
    w2v = pd(df['home_content']).values.tolist()
    for i in range(len(w2v)):
        w2v[i] = w2v[i][0].split(" ")
    return train_w2v(w2v)

def remover(text):
    text_tokens = text.split(" ")
    final_list = [word for word in text_tokens if not word in stopwords_list]
    text = ' '.join(final_list)
    return text

if __name__ == '__main__':
    df = read_csv("domain_homecontent_0to30K.csv")
    df = df[:10000]

    df[["home_content"]] = df[["home_content"]].astype(str)
    df["home_content"] = df["home_content"].apply(change_lower)
    df["home_content"] = df["home_content"].apply(clean_data)
    df["home_content"] = df["home_content"].apply(remover)

    w2v = word2vec(df)
    vector = w2v.wv.most_similar('scholarships')
    print(vector)