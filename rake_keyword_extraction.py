from rake_nltk import Rake
import argparse
import pandas as pd
import csv
import os
import re
import nltk; nltk.download('stopwords')

GAME_TOPIC_ID = {
    "alien_isolation_neg": 1,
    "alien_isolation_pos": 1,
    "fear_neg": 3,
    "fear_pos": 4,
    "hitman_neg": 0,
    "hitman_pos": 2,
    "l4d2_pos": 2,
    "shadow_of_mordor_neg": 4,
    "shadow_of_mordor_pos": 1,
}

MIN_KEYWORD_LENGTH = 2
MAX_KEYWORD_LENGTH = 3
MIN_TOPIC_PERC_CONTRIBUTION = 0.5

def clean_text(text):
    """
    Remove all non-alphabetic characters from the text.
    """
    return re.sub(r'[^A-Za-z\s]', '', text)

def get_reviews_list(topic_id, filepath, review_column='text', topic_column='dominant_topic', contribution_column='topic_perc_contrib'):
    """
    Reads a CSV file and returns all strings in the specified text column for rows 
    where the dominant topic matches the given topic_id.
    
    :param file_path: Path to the CSV file.
    :param topic_id: The topic ID to filter the rows by.
    :param text_column: Name of the column to extract text from. Default is 'text'.
    :param topic_column: Name of the column containing topic IDs. Default is 'dominant_topic'.
    :return: List of strings from the specified text column for the matching topic ID.
    """
    
    df = pd.read_csv(filepath)

    # Check if the specified columns exist in the DataFrame
    if review_column not in df.columns:
        raise ValueError(f"Column '{review_column}' does not exist in the CSV file.")
    if topic_column not in df.columns:
        raise ValueError(f"Column '{topic_column}' does not exist in the CSV file.")
    if topic_column not in df.columns:
        raise ValueError(f"Column '{contribution_column}' does not exist in the CSV file.")
    
    # Filter rows by the given topic_id
    filtered_df = df[df[topic_column] == topic_id].copy() 
    # Filter rows by the given topic_id and percent of topic contribution
    filtered_df = df[(df[topic_column] == topic_id) & (df[contribution_column] >= MIN_TOPIC_PERC_CONTRIBUTION)].copy()

    # Clean the review column
    filtered_df.loc[:, review_column] = filtered_df[review_column].apply(clean_text)
    
    # Extract the review column as a list of strings
    sentences = filtered_df[review_column].astype(str).tolist()

    return sentences

def extract_keywords(sentences):
    """
    Extract keywords from a sentences (list of strings) and returns the keyword phrases and scores as a list of tuples.
    """
    r = Rake(include_repeated_phrases=False, min_length=MIN_KEYWORD_LENGTH, max_length=MAX_KEYWORD_LENGTH)

    r.extract_keywords_from_sentences(sentences)

    data = r.get_ranked_phrases_with_scores()

    return data

def save_csv(data, file):
    """
    Get data (list of tuples (score, keyword)) and saves to CSV file. 
    """
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file,'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['score','keyword'])
        for row in data:
            csv_out.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Extract keywords from reviews on AI behaviour.')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Path to CSV files containing dominant topic by reviews.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to save CSV files containing list of scores and keywords')
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    # For each game, extract keywords and save to CSV file.
    for game_title in GAME_TOPIC_ID:
        input_file = f'{input_dir}/{game_title}_topic_sentences.csv'
        output_file = f'{output_dir}/{game_title}_extracted_keywords.csv'

        sentences = get_reviews_list(GAME_TOPIC_ID[game_title], input_file)

        data = extract_keywords(sentences)

        save_csv(data, output_file)

if __name__ == '__main__':
    main()