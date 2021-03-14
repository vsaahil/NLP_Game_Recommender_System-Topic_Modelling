# NLP_Game_Recommender_System-Topic_Modelling
This project showcases application of NLP and Topic Modelling on user and critic reviews posted on MetaCritic to generate cross-genre game recommendations. Furthermore, an additional layer of recommendation for "hidden-gems" is also developed. (Games which have lesser critic/user reviews but have generally favourable responses). Please refer to the report, presentation deck and presentation video for detailed description.

## Source of Data:
The data has been scraped from the MetaCritic website, using "1_Scrapper-Metacritic and Data Preprocessing.ipynb" file

## Methodology:

1. Scrape videogame info and reviews from Metacritic (critics and users)
2. Data Pre-processing:
• Lowercasing Reviews
• Removing Stop words, Punctuation and Game Titles
• Aggregating all reviews per game (User/Critic separately)
• Keeping Unique Game Titles
• Lemmatization
• Removing words with frequency < 50
• Removing non-essential words
3. Use TF-IDF to vectorize all text and NMF, LSA and LDA to reduce dimensionality
3. Determine recommendations using K-means Clustering algorithm with cosine similarity on the vectorized comments
4. Recommend games and “hidden-gem” games based on user and critic reviews 
[An additional layer of constraints, which only recommends games with less than 25 total number of user reviews (either favorable/mixed/average reviews); Less than 17 total number of critic reviews(either favorable/mixed/average reviews)]
