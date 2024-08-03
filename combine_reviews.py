import pandas as pd
import os
import json
from langdetect import detect, LangDetectException
import re

review_sites = ['metacritic', 'steam']
game_titles = ["l4d2", "fear", "alien_isolation", "hitman", "shadow_of_mordor", "halo_infinite", "the_last_of_us"]

def combine_reviews(output_file):
    """
    Combines reviews from multiple CSV files into a single DataFrame and saves it to a new CSV file.

    Parameters:
    output_file (str): The path to the output CSV file.

    Returns:
    pd.DataFrame: The combined DataFrame with reviews.
    """
    data_frames = []

    for game in game_titles:
        for site in review_sites:
            file_path = f"data/{site}_reviews/{game}.csv"
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                df['game_title'] = game
                df['review_site'] = site

                # Reorder columns to place 'game_title' and 'review_site' first
                cols = ['game_title', 'review_site'] + [col for col in df.columns if col not in ['game_title', 'review_site']]
                df = df[cols]

                # Filter out non-English and empty reviews
                def is_valid_review(text):
                    try:
                        return bool(re.search('[a-zA-Z]', text)) and detect(text) == 'en'
                    except LangDetectException:
                        return False

                df = df[df['review'].apply(lambda x: isinstance(x, str) and x.strip() != '' and is_valid_review(x))]
                
                data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)

    combined_df['score'] = combined_df['score'].astype(float)

    # Modify Metacritic review scores to same metric as Steam reviews
    metacritic_mask = combined_df['review_site'] == 'metacritic'
    combined_df.loc[metacritic_mask & (combined_df['score'] <= 4), 'score'] = 0
    combined_df.loc[metacritic_mask & (combined_df['score'] >= 8), 'score'] = 1
    
    # Remove Metacritic reviews that do not have scores of either 0 or 1 after modification
    combined_df = combined_df[~metacritic_mask | (combined_df['score'].isin([0, 1]))]

    combined_df.to_csv(output_file, index=False)
    return combined_df


def save_reviews_as_json(df, json_dir, summary_file):
    """
    Saves negative and positive reviews as separate JSON files and creates a summary CSV file with review statistics.

    Parameters:
    df (pd.DataFrame): DataFrame containing the reviews.
    json_dir (str): Directory where the JSON files will be saved.
    summary_file (str): Path to the CSV file where summary statistics will be saved.

    Returns:
    None
    """
    os.makedirs(json_dir, exist_ok=True)

    # Save final dataset information in separate csv file
    summary_data = []
    total_reviews = len(df)

    for game in game_titles:
        game_df = df[df['game_title'] == game]

        # Separate positive and negative reviews
        positive_reviews = game_df[game_df['score'] == 1].to_dict(orient='records')
        negative_reviews = game_df[game_df['score'] == 0].to_dict(orient='records')

        # Save positive reviews
        with open(os.path.join(json_dir, f"{game}_pos.json"), 'w', encoding='utf-8') as pos_file:
            json.dump(positive_reviews, pos_file, indent=4)

        # Save negative reviews
        with open(os.path.join(json_dir, f"{game}_neg.json"), 'w', encoding='utf-8') as neg_file:
            json.dump(negative_reviews, neg_file, indent=4)

        # Append summary information
        total_game_reviews = len(positive_reviews) + len(negative_reviews)
        summary_data.append({
            'game_title': game,
            'game_rating': 'positive',
            'num_reviews': len(positive_reviews),
            'percentage_reviews': len(positive_reviews) / total_game_reviews * 100 if total_game_reviews > 0 else 0
        })

        summary_data.append({
            'game_title': game,
            'game_rating': 'negative',
            'num_reviews': len(negative_reviews),
            'percentage_reviews': len(negative_reviews) / total_game_reviews * 100 if total_game_reviews > 0 else 0
        })

    # Create a DataFrame for the summary
    summary_df = pd.DataFrame(summary_data)
    
    # Write the summary DataFrame to a CSV file
    summary_df.to_csv(summary_file, index=False)
    
    # Add the total number of reviews as the last line in the summary CSV
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(f"\nTotal Reviews:,{total_reviews}")


def main():
    output_file = "data/game_reviews.csv"
    json_dir = "data/json_reviews_sorted_by_rating"
    data_summmary_file = "data/game_reviews_summary.csv"
    
    combined_df = combine_reviews(output_file)

    save_reviews_as_json(combined_df, json_dir, data_summmary_file)

if __name__ == "__main__":
    main()
