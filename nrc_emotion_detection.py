import pandas as pd
import os
import argparse
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import json

EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

def load_nrc_lexicon(filepath):
    """
    Loads the NRC Emotion Lexicon from a file into a dictionary.

    The NRC Emotion Lexicon file is expected to have tab-separated values with three columns:
    1. `word`: The word in the lexicon.
    2. `emotion`: The emotion associated with the word.
    3. `association`: A numerical value indicating the strength of the association (0 or 1).

    Parameters:
    filepath (str): Path to the NRC Emotion Lexicon file.

    Returns:
    dict: A dictionary where keys are words, and values are dictionaries mapping emotions to association values.
    """
    nrc_lexicon = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word, emotion, association = line.strip().split('\t')
            if word not in nrc_lexicon:
                nrc_lexicon[word] = {}
            nrc_lexicon[word][emotion] = int(association)
    return nrc_lexicon

def get_emotions_for_single_review(review, nrc_lexicon, word_emotion_counts, game_title):
    """
    Analyzes the emotions in a single review using the NRC Emotion Lexicon.

    This function tokenizes the review, maps each token to its associated emotions using the NRC Emotion Lexicon,
    and updates emotion scores and word counts for the given game title.

    Parameters:
    review (str): The text of the review to analyze.
    nrc_lexicon (dict): A dictionary where keys are words and values are dictionaries mapping emotions to association values.
    word_emotion_counts (dict): A dictionary where keys are game titles, and values are dictionaries mapping emotions to word counts.
    game_title (str): The title of the game associated with the review.

    Returns:
    dict: A dictionary mapping emotions to their aggregated association scores for the given review.
    """
    words = word_tokenize(review.lower())
    emotions = {emotion: 0 for emotion in EMOTIONS}
    
    # Store words pertaining to emotion in dict
    for word in words:
        if word in nrc_lexicon:
            for emotion, association in nrc_lexicon[word].items():
                if emotion in EMOTIONS:
                    emotions[emotion] += association
                    # If word is associated with emotion add word to dict
                    if (association == 1):
                        word_emotion_counts[game_title][emotion][word] += 1
    return emotions

def analyze_emotions(reviews_file, lexicon_file, output_file):
    """
    Analyzes emotions in reviews using the NRC Emotion Lexicon and saves the results to a CSV file.

    This function also keeps track of word counts associated with each emotion.

    Parameters:
    reviews_file (str): Path to the CSV file containing reviews. The file must include columns 'review' and 'game_title'.
    lexicon_file (str): Path to the file containing the NRC Emotion Lexicon.
    output_file (str): Path to the CSV file where the combined results (original reviews and emotion analysis) will be saved.

    Returns:
    tuple: A tuple containing:
        - pd.DataFrame: A DataFrame with combined review data and analyzed emotion scores.
        - dict: A dictionary where keys are game titles and values are dictionaries mapping emotions to word counts.
    """
    reviews_df = pd.read_csv(reviews_file)
    reviews_df['review'] = reviews_df['review'].astype(str).fillna('')
    nrc_lexicon = load_nrc_lexicon(lexicon_file)
    emotion_results = []
    word_emotion_counts = defaultdict(lambda: {emotion: Counter() for emotion in EMOTIONS})

    # Parse through reviews and count the number of words associated w/ each emotion.
    for index, row in reviews_df.iterrows():
        review = row['review']
        game_title = row['game_title']
        emotions = get_emotions_for_single_review(review, nrc_lexicon, word_emotion_counts, game_title)
        
        emotion_results.append(emotions)
    
    emotions_df = pd.DataFrame(emotion_results)
    combined_df = pd.concat([reviews_df, emotions_df], axis=1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    combined_df.to_csv(output_file, index=False)

    return combined_df, word_emotion_counts

def aggregate_emotion_counts(combined_df):
    """
    Aggregates emotion counts by game from a DataFrame.

    This function groups the DataFrame by the 'game_title' column and sums the counts of each emotion to provide
    a total count of emotion-related words for each game. It assumes that the DataFrame contains columns for each emotion.

    Parameters:
    combined_df (pd.DataFrame): A DataFrame that includes a 'game_title' column and columns for various emotions.

    Returns:
    pd.DataFrame: A DataFrame with aggregated emotion counts for each game, where each row corresponds to a game and 
                  each column corresponds to a total count of a particular emotion.
    """
    relevant_columns = ['game_title'] + EMOTIONS
    aggregated_emotions = combined_df[relevant_columns].groupby('game_title').sum().reset_index()
    
    return aggregated_emotions

def calculate_emotion_distribution_by_game(aggregate_emotions_df, output_file):
    """
    Calculates and saves the percentage distribution of emotions by game to a CSV file.

    Parameters:
    aggregate_emotions_df (pd.DataFrame): A DataFrame with aggregated emotion counts by game. It should include a 'game_title' column and columns for various emotions.
    output_file (str): Path to the CSV file where the emotion distribution percentages will be saved.

    Returns:
    None
    """
    # Calculate the total emotion counts for each game
    aggregate_emotions_df['total_emotions'] = aggregate_emotions_df[EMOTIONS].sum(axis=1)
    
    # Calculate the percentage of each emotion
    for emotion in EMOTIONS:
        aggregate_emotions_df[f'{emotion}_percentage'] = (aggregate_emotions_df[emotion] / aggregate_emotions_df['total_emotions']) * 100

    # Select only the percentage columns and the game_title
    percentage_columns = ['game_title'] + [f'{emotion}_percentage' for emotion in EMOTIONS]
    emotion_percentages = aggregate_emotions_df[percentage_columns]

    # Transpose the DataFrame
    emotion_percentages_transposed = emotion_percentages.set_index('game_title').T

    # Rename the index to 'Emotion' for clarity
    emotion_percentages_transposed.index = [emotion.replace('_percentage', '') for emotion in emotion_percentages_transposed.index]
    emotion_percentages_transposed.index.name = 'Emotion'

    # Save the transposed emotion percentages to a CSV file
    emotion_percentages_transposed.to_csv(output_file)


def save_word_emotion_counts(word_emotion_counts, json_dir):
    """
    Saves word emotion counts to JSON files for each game.

    This function processes the `word_emotion_counts` dictionary to create JSON files where each file contains the 
    frequencies of words associated with different emotions for a specific game. The words are sorted by their frequency.

    Parameters:
    word_emotion_counts (dict): A dictionary where keys are game titles and values are dictionaries mapping emotions 
                                to `Counter` objects that count the occurrences of words associated with each emotion.
    json_dir (str): Directory where the JSON files will be saved. The directory will be created if it does not exist.

    Returns:
    None
    """
    os.makedirs(json_dir, exist_ok=True)
    for game, emotions in word_emotion_counts.items():
        sorted_emotions = {}
        for emotion, counter in emotions.items():
            sorted_emotions[emotion] = dict(counter.most_common())
        json_path = os.path.join(json_dir, f'{game}_word_emotion_counts.json')
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(sorted_emotions, json_file, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Analyze emotions in game reviews using NRC Emotion Lexicon.')
    parser.add_argument('-r', '--reviews_file', type=str, required=True, help='Path to the reviews CSV file.')
    parser.add_argument('-l', '--lexicon_file', type=str, required=True, help='Path to the NRC Emotion Lexicon file.')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Path to save the emotion analysis CSV file.')
    parser.add_argument('-a', '--aggregate_file', type=str, required=True, help='Path to save the aggregate emotion counts CSV file.')
    parser.add_argument('-e', '--emotion_distribution_by_game_file', type=str, required=True, help='Path to save the emotion distribution CSV file.')
    parser.add_argument('-j', '--json_dir', type=str, required=True, help='Directory to save the word emotion count JSON files.')
    args = parser.parse_args()
    
    output_file = os.path.abspath(args.output_file)
    aggregate_file = os.path.abspath(args.aggregate_file)
    emotion_distribution_file = os.path.abspath(args.emotion_distribution_by_game_file)
    json_dir = os.path.abspath(args.json_dir)
    
    combined_df, word_emotion_counts = analyze_emotions(args.reviews_file, args.lexicon_file, output_file)
    aggregate_emotions_df = aggregate_emotion_counts(combined_df)
    aggregate_emotions_df.to_csv(aggregate_file, index=False)
    
    calculate_emotion_distribution_by_game(aggregate_emotions_df, emotion_distribution_file)
    save_word_emotion_counts(word_emotion_counts, json_dir)

if __name__ == "__main__":
    main()