import requests
from bs4 import BeautifulSoup as bs
import csv
import urllib.parse
from pathlib import Path
import logging

MAX_QUERY_NUM = 500
NUM_PER_PAGE = 100

GAME_IDS = {
    "l4d2": '550',
    "fear": '21090',
    "alien_isolation": '214490',
    "hitman": '236870',
    "shadow_of_mordor": '241930',
    "halo_infinite": '1240440',
    "the_last_of_us": '1888930'
}

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_batch_reviews(appid, cursor):
    """
    Fetches a batch of reviews for a given Steam app ID.

    Parameters:
    appid (str): The Steam app ID for which to fetch reviews.
    cursor (str): The cursor for pagination.

    Returns:
    tuple: A tuple containing a list of reviews and the next cursor for pagination.

    Raises:
    Exception: If the data retrieval fails due to a non-200 status code or a failure code in the response.
    """
    cursor = urllib.parse.quote_plus(cursor)

    url = f"http://store.steampowered.com/appreviews/{appid}?json=1&cursor={cursor}&filter=recent&language=english&review_type=all&purchase_type=all&num_per_page={NUM_PER_PAGE}"    

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data, status code: {response.status_code}")
    
    data = response.json()

    if data.get('success') != 1:
        raise Exception(f"Failed to retrieve data, success code: {data.get('success')}")
    
    reviews = data.get('reviews', [])
    cursor = data.get('cursor')

    return reviews, cursor

def get_all_reviews(appid):
    """
    Fetches all reviews for a given Steam app ID.
    
    Parameters:
    appid (str): The Steam app ID for which to fetch reviews.

    Returns:
    list: A list of dictionaries containing review data.
    """
    cursor = "*"
    reviews_list = []
    query_num = 0
    
    while (query_num < MAX_QUERY_NUM):
        review_batch, cursor = get_batch_reviews(appid, cursor)
        
        if not review_batch:
            break

        for review in review_batch:
            reviews_list.append({
                'username': review['author']['steamid'],
                'date': review['timestamp_created'],
                'score': review['voted_up'],
                'review': review['review']
            })

        if len(review_batch) < NUM_PER_PAGE:
            break

        query_num += 1

    return reviews_list

def save_to_csv(data, output_filename):
    """
    Saves a list of dictionaries to a CSV file.

    Parameters:
    data (list): A list of dictionaries containing the data to be saved.
    output_filename (str): The path to the output CSV file.

    Returns:
    None
    """

    if not data:
        logging.info("No data to save.")
        return
        
    with open(output_filename, 'w', newline='', encoding='utf-8') as data_file:
        csv_writer = csv.writer(data_file)

        header = data[0].keys()
        csv_writer.writerow(header)

        for entry in data:
            csv_writer.writerow(entry.values())
    
    logging.info(f"Data saved to {output_filename}")

def main():
    setup_logging()

    for game in GAME_IDS:

        base_data_dir = Path("data/steam_reviews")
        base_data_dir.mkdir(parents=True, exist_ok=True)
        output_filename = base_data_dir / f"{game}.csv"

        all_reviews = get_all_reviews(GAME_IDS[game])

        save_to_csv(all_reviews, output_filename)
    
if __name__ == "__main__":
    main()