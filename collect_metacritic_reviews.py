from pathlib import Path
import bs4
import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import logging

"""
IMPORTANT: Due to recent updates in `chromedriver`, you might encounter issues such as permission errors or compatibility problems.
The code may require updates to align with the latest `chromedriver`. Check the official documentation or related issues on GitHub for guidance.
"""

GAME_URLS = {
    "l4d2": "https://www.metacritic.com/game/left-4-dead-2/user-reviews/",
    "fear": "https://www.metacritic.com/game/f-e-a-r/user-reviews/",
    "alien_isolation": "https://www.metacritic.com/game/alien-isolation/user-reviews/",
    "hitman": "https://www.metacritic.com/game/hitman-3/user-reviews/",
    "shadow_of_mordor": "https://www.metacritic.com/game/middle-earth-shadow-of-mordor/user-reviews/",
    "halo_infinite": "https://www.metacritic.com/game/halo-infinite/user-reviews/",
    "the_last_of_us": "https://www.metacritic.com/game/the-last-of-us-part-ii/user-reviews/"
}

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_more_reviews(driver, pause_time=2, max_scrolls=50):
    """
    Scrolls down the page to load more reviews.

    Parameters:
    driver (webdriver.Chrome): The Selenium WebDriver instance.
    pause_time (int): Time to pause between scrolls (in seconds).
    max_scrolls (int): Maximum number of scrolls.
    """

    last_height = driver.execute_script("return document.body.scrollHeight")
    scrolls = 0

    while scrolls < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break

        last_height = new_height
        scrolls += 1

def get_game_html_data(url, filename):
    """
    Fetches HTML data for a given game URL. Downloads and saves the HTML if not already saved.

    Parameters:
    url (str): The URL of the game's review page.
    filename (str): The path to the HTML file where data will be saved.

    Returns:
    str: The HTML content of the page.
    """

    if not filename.exists():
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        driver.get(url)

        load_more_reviews(driver, pause_time=2, max_scrolls=50)

        html_data = driver.page_source

        with open(filename, "w", encoding='utf-8') as f:
            f.write(html_data)

        logging.info(f"Successfully saved HTML data to {filename}.")

        driver.quit()

    with open(filename, encoding='utf-8') as f:
        return f.read()

def extract_user_reviews(html_data):
    """
    Extracts user reviews from HTML data.

    Parameters:
    html_data (str): The HTML content as a string.

    Returns:
    list: A list of user reviews, each represented as a dictionary.
    """
    reviews_list = []

    try:
        soup = bs4.BeautifulSoup(html_data, "html.parser")
        
        # Find the header info section
        header_info = soup.find('div', class_="c-pageProductReviews_row g-outer-spacing-bottom-xxlarge")
        if not header_info:
            logging.warning("Could not find header_info section.")
            return []
        
        # Find user reviews
        user_reviews = header_info.find_all('div', class_="c-siteReview g-bg-gray10 u-grid g-outer-spacing-bottom-large")
        if not user_reviews:
            logging.warning("Could not find any user reviews.")
            return []

        for item in user_reviews:
            score_div = item.find('div', class_="c-siteReviewHeader_reviewScore")
            score = score_div.find('span').text.strip() if score_div else ""
            
            username_link = item.find('a', class_="c-siteReviewHeader_username g-text-bold g-color-gray90")
            username = username_link.text.strip() if username_link else ""
            
            date_div = item.find('div', class_="c-siteReviewHeader_reviewDate g-color-gray80 u-text-uppercase")
            date_text = date_div.text.strip() if date_div else ""
            
            review_span = item.find('div', class_="c-siteReview_quote g-outer-spacing-bottom-small").find('span')
            review_text = review_span.text.strip() if review_span else ""
            
            reviews_list.append({
                "score": score,
                "username": username,
                "date": date_text,
                "review": review_text
            })

        return reviews_list
    
    except Exception as e:
        logging.error(f"An error occurred while extracting user reviews: {e}")
        return []

def save_to_csv(data, filename):
    """
    Saves data to a CSV file.

    Parameters:
    data (list of dict): A list of dictionaries where each dictionary represents a row of data.
    filename (str): The path to the CSV file where data will be saved.
    """
    if not data:
        logging.warning("No data to save.")
        return
        
    with open(filename, 'w', newline='', encoding='utf-8') as data_file:
        csv_writer = csv.writer(data_file)

        header = ["username", "date", "score", "review"]
        csv_writer.writerow(header)

        for entry in data:
            csv_writer.writerow([entry.get("username", ""), entry.get("date", ""), entry.get("score", ""), entry.get("review", "")])

def main():
    setup_logging()

    base_data_dir = Path("data/metacritic_reviews")
    base_data_dir.mkdir(parents=True, exist_ok=True)

    for game in GAME_URLS:
        output_filename = base_data_dir / f"{game}.csv"
        html_filename = base_data_dir / f"{game}.html"

        logging.info(f"Processing reviews for {game}...")

        try:
            html_data = get_game_html_data(GAME_URLS[game], html_filename)
            review_info = extract_user_reviews(html_data)

            if review_info:
                save_to_csv(review_info, output_filename)
                logging.info(f"Successfully saved reviews for {game} to {output_filename}.")
            else:
                logging.info(f"No user reviews found for {game}.")

        except Exception as e:
            logging.error(f"Error processing {game}: {e}")

if __name__ == "__main__":
    main()

