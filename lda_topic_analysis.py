import nltk; #nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
import os
import json

# Genism
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
from gensim.models.coherencemodel import CoherenceModel

# Spacy
import spacy

# Plot
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

# Logging for gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stopwords
from nltk.corpus import stopwords

GAME_TITLES = ["l4d2", "fear", "alien_isolation", "hitman", "shadow_of_mordor", "halo_infinite", "the_last_of_us"]
GAME_RATINGS = ["pos", "neg"]

NUM_TOPICS = 5 # number of topics to model LDA
ALPHA = 0.5
REPRESENTATIVE_DOC_NUM = 10 # number of documents (reviews) to extract to represent each topic

def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def write_data(file, data):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    texts_out = []
    for text in texts:
        # Remove non-English characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        doc = nlp(text)
        new_text = []
        # iterate over each word in doc
        for token in doc:
            # reduce complexity of text by lemmatizing word
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_) # append lemmatized form
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

def tokenize_text(reviews):
    final = []
    for review in reviews:
        new = gensim.utils.simple_preprocess(review, deacc=True) # deacc to normalize text containing accents
        final.append(new)
    return(final)

def remove_stopwords(texts):
    stop_words = set(stopwords.words('english'))
    stop_words.update(['from', 'subject', 're', 'edu', 'use', 'game', 'good', 'get', 'play'])
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram):
    return([bigram[doc] for doc in texts])

def make_trigrams(texts, bigram, trigram):
    return([trigram[bigram[doc]] for doc in texts])

def save_lda_scores(game, rating, perplexity, coherence, score_file):
    # Check if the score file exists
    if not os.path.exists(score_file):
        # Create the DataFrame and save it to a new CSV file
        df = pd.DataFrame(columns=["game", "rating", "perplexity", "coherence"])
        df.to_csv(score_file, index=False)

    # Load the existing scores
    df = pd.read_csv(score_file)

    # Append the new scores
    new_row = {"game": game, "rating": rating, "perplexity": perplexity, "coherence": coherence}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save back to CSV
    df.to_csv(score_file, index=False)


def get_dominant_topic_by_review(lda_model, corpus, texts):
    sent_topics_list = []

    # Get main topic for each document
    for i, row in enumerate(lda_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the dominant topic, percent contribution, and keywords for each doc
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_list.append([int(topic_num), round(prop_topic, 4), topic_keywords])
            else:
                break

    # Create DataFrame from the list of rows
    sent_topics_df = pd.DataFrame(sent_topics_list, columns=['dominant_topic', 'perc_contribution', 'keywords'])

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df

def get_representative_docs_by_topic(topic_sentences_df):
    representative_docs_df = pd.DataFrame()

    topic_sentences_df_grpd = topic_sentences_df.groupby('dominant_topic')
    
    for i, grp in topic_sentences_df_grpd:
        representative_docs_df = pd.concat([representative_docs_df, 
                                             grp.sort_values(['perc_contribution'], ascending=[0]).head(REPRESENTATIVE_DOC_NUM)], 
                                            axis=0)
    
    representative_docs_df.reset_index(drop=True, inplace=True)
    representative_docs_df.columns = ['topic_num', "topic_perc_contrib", "keywords", "text"]

    return representative_docs_df

def get_all_topics(lda_model):
    all_topics = []
    
    for topic_id in range(NUM_TOPICS):
        topic_keywords = [word for word, prop in lda_model.show_topic(topic_id)]
        all_topics.append(topic_keywords)
    
    return all_topics


def get_coherence_score_per_topic(lda_model, dictionary, texts):
    # Get topic list
    topics = get_all_topics(lda_model)

    # Initialize the CoherenceModel
    coherence_model = CoherenceModel(topics=topics, model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    
    # Get the coherence score for each topic
    coherence_list = coherence_model.get_coherence_per_topic()
    
    # Create a DataFrame for the topic coherences and keywords
    df_topic_coherences = pd.DataFrame({
        'Coherence': coherence_list,
        'Keywords': topics
    })
    
    return df_topic_coherences

def model_lda(game, rating, input_dir, output_dir):
    df = load_data(f"{input_dir}/{game}_{rating}.json")
    
    # Filter out non-string entries and handle NaN values
    data = df['review'].dropna().astype(str).tolist()

    # PREPROCESS REVIEWS
    lemmatized_texts = lemmatization(data)

    data_words = tokenize_text(lemmatized_texts)

    data_words = remove_stopwords(data_words)

    # BIGRAMS AND TRIGRAMS
    bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)

    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = make_bigrams(data_words, bigram)
    data_bigrams_trigrams = make_trigrams(data_bigrams, bigram, trigram)

    # Create dictionary and corpus
    id2word = corpora.Dictionary(data_bigrams_trigrams)
    texts = data_bigrams_trigrams

    corpus = [id2word.doc2bow(text) for text in texts]

    # TF-IDF FILTER
    tfidf = TfidfModel(corpus, id2word=id2word)

    low_value = 0.03
    words  = []
    words_missing_in_tfidf = []
    for i in range(0, len(corpus)):
        bow = corpus[i]
        low_value_words = []
        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words+words_missing_in_tfidf
        for item in drops:
            words.append(id2word[item])
        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf score 0 will be missing

        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
        corpus[i] = new_bow
    
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=NUM_TOPICS,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha=0.5)
    
    
    # Save model to disk
    path_to_save_models = f'{output_dir}/lda_models'
    os.makedirs(path_to_save_models, exist_ok=True)
    lda_model.save(f'{path_to_save_models}/{game}_{rating}_lda.model')

    # Save dictionary
    path_to_dict = f'{output_dir}/lda_dicts'
    os.makedirs(path_to_dict, exist_ok=True)
    id2word.save(f'{path_to_dict}/{game}_{rating}_dict.dict')

    # Save corpus
    path_to_corpus = f'{output_dir}/lda_corpus'
    os.makedirs(path_to_corpus, exist_ok=True)
    corpora.MmCorpus.serialize(f'{path_to_corpus}/{game}_{rating}_corpus.mm', corpus)

    # Visualize data
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)

    # Save html data
    HTLML_DATA_DIR = f'{output_dir}/html_data'
    os.makedirs(HTLML_DATA_DIR, exist_ok=True)
    pyLDAvis.save_html(vis, f"{HTLML_DATA_DIR}/{game}_{rating}_lda.html")

    # Finding the dominant topic for each review
    dominant_topic_by_review_dir = f"{output_dir}/dominant_topic_by_review"
    os.makedirs(dominant_topic_by_review_dir, exist_ok=True)
    topic_sentences_df = get_dominant_topic_by_review(lda_model, corpus, data)

    df_dominant_topic = topic_sentences_df.reset_index()
    df_dominant_topic.columns = ['document_no', 'dominant_topic', 'topic_perc_contrib', 'keywords', 'text']

    df_dominant_topic.to_csv(f"{dominant_topic_by_review_dir}/{game}_{rating}_topic_sentences.csv", index=False)

    # Save top most representative document for each topic
    representative_docs_dir = f"{output_dir}/representative_docs"
    os.makedirs(representative_docs_dir, exist_ok=True)

    representative_docs_df = get_representative_docs_by_topic(topic_sentences_df)
    representative_docs_df.to_csv(f"{representative_docs_dir}/{game}_{rating}_representative_docs.csv")

    # Save coherence scores for each topic
    topic_coherences_dir = f"{output_dir}/topics_sorted_by_coherence"
    os.makedirs(topic_coherences_dir, exist_ok=True)
    topic_coherences_dir = os.path.join(topic_coherences_dir, f"{game}_{rating}_topic_coherence.csv")

    topic_coherences_df = get_coherence_score_per_topic(lda_model, id2word, texts)
    topic_coherences_sorted_df = topic_coherences_df.sort_values(by='Coherence', ascending=False)
    
    topic_coherences_sorted_df.to_csv(topic_coherences_dir, index=False)


    # Compute perplexity and topic coherence for LDA model
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_bigrams_trigrams, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    return lda_model.log_perplexity(corpus), coherence_lda

def main():
    # Path to reviews in json format
    input_dir = "data/json_reviews_sorted_by_rating"
    
    # Path to save lda data visualization
    output_dir = f"data/lda"

    # Path to save coherence scores
    score_file = f"{output_dir}/lda_scores.csv"

    for game in GAME_TITLES:
        for rating in GAME_RATINGS:
            perplexity, coherence = model_lda(game, rating, input_dir, output_dir)
            save_lda_scores(game, rating, perplexity, coherence, score_file)

if __name__ == "__main__":
    main()