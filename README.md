# AI-VideoGames-Comparative-Analysis

## Overview

This repository contains the code and data associated with the research paper titled **"A Comparative Study of AI in Video Games via Sentiment Analysis and Topic Modeling of User Reviews"** by Seunghyun Nam. The project is designed to analyze user reviews for video games that utilize AI, specifically focusing on the titles **Alien: Isolation**, **F.E.A.R.**, **Halo Infinite**, **Hitman**, **Left 4 Dead 2**, **Shadow of Mordor**, and **The Last of Us**.

### Features

- **Review Scraping**: Extracts user reviews from Metacritic and Steam.
- **Emotion Analysis**: Utilizes the NRC word-emotion lexicon to perform sentiment analysis on the reviews.
- **Topic Modeling**: Applies Latent Dirichlet Allocation (LDA) to identify topics within the reviews.
- **Keyword Extraction**: Employs the RAKE algorithm for keyword extraction from reviews with specific topics.

### Customization

The provided scripts can be easily adapted to analyze different video games available on Metacritic or Steam, and adjust parameters to fit specific analysis needs. 

## Installation

Instructions on how to set up and install the project.

## Running the Scripts

### 1. Scraping Reviews from Steam and Metacritic

To collect reviews from both Steam and Metacritic, run the following scripts:

- **`collect_metacritic_reviews.py`**: Scrapes user reviews from Metacritic.
- **`collect_steam_reviews.py`**: Scrapes user reviews from Steam.

#### Configuring Game Lists

- **Metacritic**: Open `collect_metacritic_reviews.py` and update the `GAME_URLS` dictionary on line 15 with your list of games and their corresponding URLs.
- **Steam**: Open `collect_steam_reviews.py` and update the `GAME_IDS` variable on line 11 with a dictionary containing game titles and their Steam IDs.

### 2. Combining Reviews

Use the script **`combine_reviews.py`** to merge reviews from both Metacritic and Steam into a single file.
Update the `game_titles` variable in `combine_reviews.py` on line 8 with your own list of game titles.

#### Output Files

This script will generate the following files in the `data` directory:
- **`game_reviews.csv`**: Contains all the collected reviews.
- **`json_reviews_sorted_by_rating`**: A subdirectory with two JSON files for each game (positive and negative reviews).
- **`game_reviews_summary.csv`**: Summarizes the total number and percentages of positive and negative reviews for each game after filtering.

#### Note

The script modifies Metacritic review scores on a 1-10 scale to a binary scale (0 and 1), aligning them with Steam’s review metric. Scores in the range of 5-8 are filtered out. To retain original scores, remove lines 51-57 from the script.

### 3. NRC Emotion Analysis

Use the **`nrc_emotion_detection.py`** script to conduct sentiment analysis using the NRC emotion lexicon. This script scores the words in the reviews according to the emotions associated with them.

#### Steps:

1. **Download the NRC Emotion Lexicon**:
   - Download the lexicon from [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm).
   
2. **Run the Script**:
   - Execute `nrc_emotion_detection.py` with the appropriate arguments for input and output files.

   Example command:
   ```bash
   python nrc_emotion_detection.py -r game_reviews.csv -l NRC-Emotion-Lexicon-Wordlevel-v0.92.txt -o nrc_emotion_analysis_results/nrc_emotion_analysis.csv -a nrc_emotion_analysis_results/nrc_aggregate_emotion_count_by_game.csv -e nrc_emotion_analysis_results/nrc_emotion_distribution_by_game.csv -j emotion_words_sorted_by_frequency

#### Output Files:

The script generates the following files:

- **`--output_file`**: A CSV file displaying the number of words associated with each emotion for each review.
- **`--aggregate_file`**: A CSV file showing the aggregate count of words associated with each emotion by game.
- **`--emotion_distribution_by_game_file`**: A CSV file detailing the emotional distribution for each game based on the aggregate counts.
- **`--json_dir`**: A subdirectory containing JSON files with dictionaries of emotions and associated words, sorted by frequency.

### 4. LDA Topic Analysis

Use the **`lda_topic_analysis.py`** script to perform LDA topic analysis on the reviews for each game. The script performs separate topic analyses for positive and negative reviews, utilizing files in the `json_reviews_sorted_by_rating` subdirectory. It employs `gensim` for the LDA model and `pyLDAvis` for visualization.

### Parameters:

- **Change Games**: Modify the `GAME_TITLES` variable in line 34 to update the list of games.
- **Number of Topics**: Adjust the `NUM_TOPICS` variable in line 37 to set the number of topics.
- **Alpha Parameter**: Adjust the `ALPHA` variable in line 38 to modify the alpha parameter for the LDA model.
- **Representative Reviews**: Set the number of representative reviews to display by modifying `REPRESENTATIVE_DOC_NUM` in line 39.

### Output Files:

The script outputs the following in the `data` directory:

- **`dominant_topic_by_review`**: A subdirectory containing a CSV file for each game, with separate files for positive and negative reviews. Each CSV file includes rows of reviews and the dominant topic ID for each review (i.e., the topic that contributed the most or had the highest number of tokens).

- **`representative_docs`**: A subdirectory containing files for each topic. Each file lists the top 10 reviews with the highest percentage contribution of that topic. This helps in understanding the context of each topic, especially if the topics are unclear from the keywords alone.

- **`topics_sorted_by_coherence`**: A subdirectory with a CSV file for each game, showing the coherence score and top keywords for each topic.

- **`html_data`**: A subdirectory containing HTML files for each game that visualize the topic distribution using `pyLDAvis`.

- **`lda_corpus`, `lda_dicts`, `lda_models`**: These subdirectories store the LDA corpus, dictionary, and models used during analysis.

- **`lda_scores.csv`**: A CSV file containing the perplexity and coherence scores for each game.

### 5. RAKE Keyword Analysis

The **`rake_keyword_extraction.py`** script is used to perform keyword extraction using the RAKE (Rapid Automatic Keyword Extraction) algorithm, utilizing the `rake-nltk` library. This script focuses on extracting keywords from reviews for select games that had "AI" as one of the five topics in the LDA analysis. Specifically, it targets reviews where "AI" was the dominant topic.

The reviews are filtered based on the `GAME_TOPIC_ID`, which corresponds to the topic ID from the `dominant_topic` column in the `dominant_topic_by_review` files.

**NOTE**: The dominant topic ID is different from the topic number shown in the pyLDAvis maps in the HTML files, which sort topics based on token prevalence. Modify the `GAME_TOPIC_ID` in line 9 to match your list of games and topic.

#### Parameters:

- **To specify keyword lengths**: Change the following parameters in lines 21-23:
  - `MIN_KEYWORD_LENGTH`: Minimum length of keywords to extract. The default length is 2.
  - `MAX_KEYWORD_LENGTH`: Maximum length of keywords to extract. The default length is 3.
- `MIN_TOPIC_PERC_CONTRIBUTION`: Minimum percentage contribution of a topic for a review to be included. The default threshold is 0.5.

This configuration allows you to adjust the granularity of keyword extraction and the minimum contribution required for reviews to be considered in the analysis.

## Known Issues

### ChromeDriver Issues in `collect_metacritic_reviews.py`

Due to recent updates in `chromedriver`, there have been issues with installing and executing the driver.

**If you encounter issues like:**
- `[Errno 8] Exec format error`
- `Driver [THIRD_PARTY_NOTICES.chromedriver] found in cache`

**Please note:**
- Ensure that you have the latest version of `webdriver_manager`.
- Remove the `.wdm` folder and then reinstall `webdriver-manager`.
- Replace the file `THIRD_PARTY_NOTICES.chromedriver` with `chromedriver`.

**Code Update:**
The code may need updates to align with the latest version of `chromedriver`. 

## Contact
Seunghyun Nam - shyunam@gmail.com
