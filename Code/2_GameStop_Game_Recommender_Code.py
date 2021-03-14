# NLP_Game_Recommender_System-Topic_Modelling

# Projct Members:
# 1. Jawad Toufaili 
# 2. Sebastian Salazar 
# 3. Shivangi Soni 
# 4. Vivek Saahil 

#------------------------------------------------------------------------------------------------

# Importing Libraries
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import nltk
import string
import textblob
import warnings
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn import manifold
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentiText
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
warnings.filterwarnings('ignore')
from wordcloud import WordCloud, STOPWORDS 

#pip install -U textblob
#pip install langdetect
#pip install vaderSentiment
#pip install wordcloud
# Run the above in Jupyter Notebook

#------------------------------------------------------------------------------------------------

# Loading Data
            
# Loading user_reviews
user_reviews = pd.read_csv(r"C:\\Users\\vivek\\Desktop\\switch_user_review_filtered.csv")

# Loading critic_reviews
critic_reviews = pd.read_csv(r"C:\\Users\\vivek\\Desktop\\switch_critic_review_filtered.csv")

# Loading sales data
sales = pd.read_csv(r"C:\\Users\\vivek\\Desktop\\sales_switch.csv")

# Loading game information
game_info = pd.read_csv(r"C:\\Users\\vivek\\Desktop\\switch_game_info_filtered.csv")

#------------------------------------------------------------------------------------------------

# Getting unique games
def game_id_df_calc(data):
    df_pivot = pd.pivot_table(data, values = ['score'], index = ['title', 'platform'])
    unique_games_platform_list = df_pivot.index
    
    game_id_list = []
    game_list = []
    platform_list = []
    
    game_id = 0
    for unique_game in unique_games_platform_list:
        game_id += 1
        game_id_list.append(game_id)
        game_list.append(unique_game[0])
        platform_list.append(unique_game[1])
        
    game_id_df = pd.DataFrame()
    game_id_df['game_id'] = game_id_list
    game_id_df['title'] = game_list
    game_id_df['platform'] = platform_list
    return game_id_df

game_id_df1 = game_id_df_calc(user_reviews)
game_id_df2 = game_id_df_calc(critic_reviews)

#------------------------------------------------------------------------------------------------

# For user_reviews

# Data Pre-processing
df = user_reviews

# Extracting titles (ID) of all games
game_title_list = df['title'].tolist()

# Lowercasing Reviews
word_comments=[]
for i in range(len(df)):
    word_comments.append(df['text'][i].lower())
df['text_lower'] = word_comments

# Removing stopwords 
stop = nltk.corpus.stopwords.words('english')

# Removing Game Titles
game_titles = list(df['title'].unique())
lowercase_game_titles = [title.lower().split(': ') for title in game_titles]

titles_to_remove = []
for title in lowercase_game_titles:
    if len(title) == 2:
        titles_to_remove.append(title[0])
        titles_to_remove.append(title[1])
    else:
        titles_to_remove.append(title[0])

stop.extend(titles_to_remove)

# Stop word Dictionary
stopword_dict = {}

for stopword in stop:
    stopword_dict[stopword] = 1

df['cleaned_text'] = df['text_lower'].apply(lambda x: " ".join(x for x in x.split() if x not in stopword_dict))

# Aggregating all comments per game
game_id_comment_list = []
comments_for_games_list = []

for i in game_title_list:
    review_text = df[df['title'] == i]['cleaned_text'].str.cat(sep=' ').strip(' ').lower()
    game_id_comment_list.append(i)
    comments_for_games_list.append(review_text)

comments_for_games_df = pd.DataFrame()
comments_for_games_df['title'] = game_id_comment_list
comments_for_games_df['reviews'] = comments_for_games_list

# Removing Punctuation
comments_for_games_df['reviews'] = comments_for_games_df['reviews'].str.replace('[^\w\s]',' ')

# Keeping Unique title values
final_df = comments_for_games_df.drop_duplicates(subset = ["title"])
final_df

final_df1 = final_df

# Final Dataframe to be used -> final_df1

#------------------------------------------------------------------------------------------------

word_frequency = pd.Series(' '.join(final_df1['reviews']).split()).value_counts()
word_frequency[0:100]

# Remove words that occur less than 50 times
#import matplotlib.pyplot as plt
#plt.hist(word_frequency)
# Total words: 41,180
count = 0
to_remove_list = []

word_frequency = word_frequency.reset_index()
word_frequency = word_frequency.rename(columns={"index": "words", 0: "frequency"})

for i in range(0,len(word_frequency)):
    if word_frequency['frequency'][i]<=50:
        count = count + 1
        to_remove_list.append(word_frequency['words'][i])
count
# to_remove_list = to_remove_list[0]
# To remove 38,918 words   
        
stop.extend(to_remove_list)

to_remove_words =['game', 'games',"'ll", "una", "40", "ha", "ca","wa", "se", "zelda", "este", "del", "'s", "n't", "...", '"', "'ve", "``", "'", "'m", "'re", "'d", "si", "30", "9", "te", "60", 'de', 'switch', 'nintendo', 'i', 'que', 'el', 'en', 'es', '10', 'la', 't','un', '2', 'juego', 've', 'm', '3', '1', '5', 'lo', '8', '4', 'd', 'u', '100', 'las','6', '7','co', 'op', 'su']
stop.extend(to_remove_words)


stopword_dict = {}

for stopword in stop:
    stopword_dict[stopword] = 1

final_df1['reviews'] = final_df1['reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stopword_dict))

# final_df_wo_lemma = final_df1

# Lemmatization
from textblob import Word
final_df1['reviews'] = final_df1['reviews'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#------------------------------------------------------------------------------------------------

#Topic Modeling

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors

train_data_reviews = final_df1['reviews']

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),  
                                   stop_words='english', 
                                   token_pattern="\\b[a-z][a-z]+\\b",
                                   lowercase=True,
                                   max_df = 0.6)

tfidf_data = tfidf_vectorizer.fit_transform(train_data_reviews)

def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Display topics for NMF on TF-IDF Vectorizer
nmf = NMF(n_components = 25)
nmf_tfidf_data = nmf.fit_transform(tfidf_data)
#display_topics(nmf,tfidf_vectorizer.get_feature_names(),10)

# Display topics for LSA on TF-IDF Vectorizer
lsa = TruncatedSVD(n_components = 25)
lsa_tfidf_data = lsa.fit_transform(tfidf_data)
#display_topics(lsa,tfidf_vectorizer.get_feature_names(),10)

# Display topics for LDA on TF-IDF Vectorizer
lda = LatentDirichletAllocation(n_components=25)
lda_tfidf_data = lda.fit_transform(tfidf_data)
#display_topics(lda,tfidf_vectorizer.get_feature_names(),10)

#------------------------------------------------------------------------------------------------

# Clustering
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# NMF TF-IDF
ssx_nmf_tfidf = StandardScaler().fit_transform(nmf_tfidf_data)

# LSA TF-IDF
ssx_lsa_tfidf = StandardScaler().fit_transform(lsa_tfidf_data)

# LDA TF-IDF
ssx_lda_tfidf = StandardScaler().fit_transform(lda_tfidf_data)

# K-Means & SSE
def get_cluster_centers(X, labels, k_num):
    CC_list = []
    for k in range(k_num):
        CC_list.append(np.mean(X[labels == k], axis = 0))
    return CC_list

def get_SSE(X, labels):
    k_num = len(np.unique(labels))
    CC_list = get_cluster_centers(X, labels, k_num)
    CSEs = []
    for k in range(k_num):
        error_cords = X[labels == k] - CC_list[k]
        error_cords_sq = error_cords ** 2
        error_mag_sq = np.sum(error_cords_sq, axis = 1)
        CSE = np.sum(error_mag_sq)
        CSEs.append(CSE)
    return sum(CSEs)

def get_silhouette_sse(vectorized_data, cluster_range):
    Sil_coefs = []
    SSEs = []
    
    for k in cluster_range:
        km = KMeans(n_clusters=k, random_state=25)
        km.fit(vectorized_data)
        labels = km.labels_
        Sil_coefs.append(silhouette_score(vectorized_data, labels, metric='euclidean'))
        SSEs.append(get_SSE(vectorized_data, labels))
    return cluster_range, Sil_coefs, SSEs

'''
# Clustering function
def clusterer(model_transformer,name):
    k_clusters, silhouette_coefs, sse = get_silhouette_sse(model_transformer, range(2,30))
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5), sharex=True)
    
    ax1.plot(k_clusters, silhouette_coefs)
    ax1.set_title(name+' with TF-IDF Silhouette Coefficient')
    ax1.set_xlabel('number of clusters')
    ax1.set_ylabel('silhouette coefficient')
    plt.xticks(np.arange(2, 30, step=2))
    
    # plot Inertia/SSE on ax2
    ax2.plot(k_clusters, sse)
    ax2.set_title(name+' with TF-IDF SSE')
    ax2.set_xlabel('number of clusters')
    ax2.set_ylabel('SSE');    

# Clustering - NMF TF-IDF
clusterer(ssx_nmf_tfidf,'NMF')

# Clustering - LSA TF-IDF
clusterer(ssx_lsa_tfidf,'LSA')

# Clustering - LDA TF-IDF
clusterer(ssx_lda_tfidf,'LDA')
'''

#------------------------------------------------------------------------------------------------

# Recommender based on user reviews
def user_recommend(game_name):
    data_index = final_df1[final_df1['title'] == game_name].index[0]
    train_data_reviews[data_index]
        
    new_datapoint = [train_data_reviews[data_index]]
    new_datapoint
    new_vec = nmf.transform(tfidf_vectorizer.transform(new_datapoint))
        
    nn = NearestNeighbors(n_neighbors = 50, metric='cosine', algorithm='brute')
    nn.fit(nmf_tfidf_data)
    result = nn.kneighbors(new_vec)
    result[1][0]
    
    # Generating recommendation list
    recommendation_list=[]
    for r in result[1][0]:
        game = game_id_df1.title[r]
        if game != game_name:
            recommendation_list.append(game)

    return recommendation_list

#------------------------------------------------------------------------------------------------

# Merging sales table
game_id_df1_w_sales = pd.merge(game_id_df1, sales, on='title', how='left')
game_id_df1_merged = pd.merge(game_id_df1_w_sales, game_info, on='title', how='left')
game_id_df1_merged1 = game_id_df1_merged[['game_id','title','total_sales_USDMM','meta_overview','user_overview']]

#--------------------------------------------------------------------------------------
# Getting hidden gems 

# Getting sum of reviews 
game_info['user_pos'] = pd.to_numeric(game_info['user_pos'])
game_info['user_neg']=pd.to_numeric(game_info['user_neg'])
game_info['sum_reviews']= game_info.iloc[:,13:16].sum(axis=1)

# Threshold for defining hidden gems
count = 0 
for i in range(len(game_info)):
    if (game_info.sum_reviews[i] <=25):
        count +=1
print(count)

# Hidden gems = games with comments of less than 25(inclusive) and the ones with generally favorable and mixed or average reviews 
# Making a list of hidden gems 
hidden_df = game_info[['title', 'meta_overview','user_overview', 'sum_reviews']]    
hidden_df = hidden_df[hidden_df['sum_reviews'] <=25]
hidden_df = hidden_df[(hidden_df['user_overview'] == 'Generally favorable reviews') | (hidden_df['user_overview'] == 'Mixed or average reviews')]

# Total of 103 underexposed games that have generally favorable reviews but lower than 25 comments 

# Checking for how many do we have sales data 
hidden_sales = pd.merge(hidden_df,sales, on='title', how='left')
#*Very few have sales data so not using sales data 

# Merging hidden sales id with the game id
hidden_id = pd.merge(hidden_df,game_id_df1, on='title', how='left')

# Dropping the games that  didn't have game_id
hidden_id = hidden_id.dropna(axis=0)

# Creating a list for hidden gems that can be showcased to people 
hidden_list = hidden_sales['title'].to_list()

hidden_id_list = (hidden_id['game_id']).astype(int).to_list()

# Hidden gems recommender
def hidden_user_recommend(game_name):
    data_index1 = final_df1[final_df1['title'] == game_name].index[0]
    train_data_reviews[data_index1]
        
    new_datapoint1 = [train_data_reviews[data_index1]]
    new_vec1 = nmf.transform(tfidf_vectorizer.transform(new_datapoint1))
        
    nn1 = NearestNeighbors(n_neighbors = 100, metric='cosine', algorithm='brute')
    nn1.fit(nmf_tfidf_data)
    result1 = nn1.kneighbors(new_vec1)
    result1[1][0]
    
        
    # Generating recommendation list
    recommendation_list1=[]
    for r in result1[1][0]:
        if r in hidden_id_list:
            game = game_id_df1.title[r]
            if game != game_name:
                recommendation_list1.append(game)
    return recommendation_list1

#------------------------------------------------------------------------------------------------






#------------------------------------------------------------------------------------------------

# For critic_reviews

# Data Pre-processing
df_critic = critic_reviews

# Extracting titles (ID) of all games

game_title_list = df_critic['title'].tolist()

# Lowercasing Reviews
word_comments=[]
for i in range(len(df_critic)):
    word_comments.append(df_critic['text'][i].lower())
df_critic['text_lower'] = word_comments

# Removing stopwords 
stop = nltk.corpus.stopwords.words('english')

# Removing Game Titles
game_titles = list(df_critic['title'].unique())
lowercase_game_titles = [title.lower().split(': ') for title in game_titles]

titles_to_remove = []
for title in lowercase_game_titles:
    if len(title) == 2:
        titles_to_remove.append(title[0])
        titles_to_remove.append(title[1])
    else:
        titles_to_remove.append(title[0])

stop.extend(titles_to_remove)


# Stop word Dictionary
stopword_dict = {}

for stopword in stop:
    stopword_dict[stopword] = 1

df_critic['cleaned_text'] = df_critic['text_lower'].apply(lambda x: " ".join(x for x in x.split() if x not in stopword_dict))

# Aggregating all comments per game
game_id_comment_list = []
comments_for_games_list = []

for i in game_title_list:
    review_text = df_critic[df_critic['title'] == i]['cleaned_text'].str.cat(sep=' ').strip(' ').lower()
    game_id_comment_list.append(i)
    comments_for_games_list.append(review_text)

comments_for_games_df_critic = pd.DataFrame()
comments_for_games_df_critic['title'] = game_id_comment_list
comments_for_games_df_critic['reviews'] = comments_for_games_list

# Removing Punctuation
comments_for_games_df_critic['reviews'] = comments_for_games_df_critic['reviews'].str.replace('[^\w\s]',' ')

# Keeping Unique title values
final_df_critic = comments_for_games_df_critic.drop_duplicates(subset = ["title"])
final_df_critic

final_df1_critic = final_df_critic

# Final Dataframe to be used -> final_df1_critic

#------------------------------------------------------------------------------------------------

word_frequency = pd.Series(' '.join(final_df1_critic['reviews']).split()).value_counts()
word_frequency[0:100]

# Remove words that occur less than 50 times
#import matplotlib.pyplot as plt
#plt.hist(word_frequency)
# Total words: 41,180
count = 0
to_remove_list = []

word_frequency = word_frequency.reset_index()
word_frequency = word_frequency.rename(columns={"index": "words", 0: "frequency"})

for i in range(0,len(word_frequency)):
    if word_frequency['frequency'][i]<=50:
        count = count + 1
        to_remove_list.append(word_frequency['words'][i])
count
# to_remove_list = to_remove_list[0]
        
stop.extend(to_remove_list)

to_remove_words =['game', 'games',"'ll", "una", "40", "ha", "ca","wa", "se", "zelda", "este", "del", "'s", "n't", "...", '"', "'ve", "``", "'", "'m", "'re", "'d", "si", "30", "9", "te", "60", 'de', 'switch', 'nintendo', 'i', 'que', 'el', 'en', 'es', '10', 'la', 't','un', '2', 'juego', 've', 'm', '3', '1', '5', 'lo', '8', '4', 'd', 'u', '100', 'las','6', '7','co', 'op', 'su']
stop.extend(to_remove_words)

stopword_dict = {}

for stopword in stop:
    stopword_dict[stopword] = 1

final_df1_critic['reviews'] = final_df1_critic['reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stopword_dict))

# final_df_wo_lemma_critic = final_df1_critic

# Lemmatization
from textblob import Word
final_df1_critic['reviews'] = final_df1_critic['reviews'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#------------------------------------------------------------------------------------------------

#Topic Modeling & Building Game Recommender by Text

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
#from gensim.models import word2vec

train_data_reviews_critic = final_df1_critic['reviews']

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),  
                                   stop_words='english', 
                                   token_pattern="\\b[a-z][a-z]+\\b",
                                   lowercase=True,
                                   max_df = 0.6)

tfidf_data = tfidf_vectorizer.fit_transform(train_data_reviews_critic)

# Topic Modeling:
def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Display topics for NMF on TF-IDF Vectorizer
nmf = NMF(n_components = 25)
nmf_tfidf_data = nmf.fit_transform(tfidf_data)
#display_topics(nmf,tfidf_vectorizer.get_feature_names(),10)

# Display topics for LSA on TF-IDF Vectorizer
lsa = TruncatedSVD(n_components = 25)
lsa_tfidf_data = lsa.fit_transform(tfidf_data)
#display_topics(lsa,tfidf_vectorizer.get_feature_names(),10)

# Display topics for LDA on TF-IDF Vectorizer
lda = LatentDirichletAllocation(n_components=25)
lda_tfidf_data = lda.fit_transform(tfidf_data)
#display_topics(lda,tfidf_vectorizer.get_feature_names(),10)

#------------------------------------------------------------------------------------------------

# Clustering
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# NMF TF-IDF
ssx_nmf_tfidf = StandardScaler().fit_transform(nmf_tfidf_data)

# LSA TF-IDF
ssx_lsa_tfidf = StandardScaler().fit_transform(lsa_tfidf_data)

# LDA TF-IDF
ssx_lda_tfidf = StandardScaler().fit_transform(lda_tfidf_data)

# K-Means & SSE
def get_cluster_centers(X, labels, k_num):
    CC_list = []
    for k in range(k_num):
        CC_list.append(np.mean(X[labels == k], axis = 0))
    return CC_list

def get_SSE(X, labels):
    k_num = len(np.unique(labels))
    CC_list = get_cluster_centers(X, labels, k_num)
    CSEs = []
    for k in range(k_num):
        error_cords = X[labels == k] - CC_list[k]
        error_cords_sq = error_cords ** 2
        error_mag_sq = np.sum(error_cords_sq, axis = 1)
        CSE = np.sum(error_mag_sq)
        CSEs.append(CSE)
    return sum(CSEs)

def get_silhouette_sse(vectorized_data, cluster_range):
    Sil_coefs = []
    SSEs = []
    
    for k in cluster_range:
        km = KMeans(n_clusters=k, random_state=25)
        km.fit(vectorized_data)
        labels = km.labels_
        Sil_coefs.append(silhouette_score(vectorized_data, labels, metric='euclidean'))
        SSEs.append(get_SSE(vectorized_data, labels))
    return cluster_range, Sil_coefs, SSEs

'''
# Clustering function
def clusterer(model_transformer,name):
    k_clusters, silhouette_coefs, sse = get_silhouette_sse(model_transformer, range(2,30))
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5), sharex=True)
    
    ax1.plot(k_clusters, silhouette_coefs)
    ax1.set_title(name+' with TF-IDF Silhouette Coefficient')
    ax1.set_xlabel('number of clusters')
    ax1.set_ylabel('silhouette coefficient')
    plt.xticks(np.arange(2, 30, step=2))
    
    # plot Inertia/SSE on ax2
    ax2.plot(k_clusters, sse)
    ax2.set_title(name+' with TF-IDF SSE')
    ax2.set_xlabel('number of clusters')
    ax2.set_ylabel('SSE');    

# Clustering - NMF TF-IDF
clusterer(ssx_nmf_tfidf,'NMF')

# Clustering - LSA TF-IDF
clusterer(ssx_lsa_tfidf,'LSA')

# Clustering - LDA TF-IDF
clusterer(ssx_lda_tfidf,'LDA')
'''

#------------------------------------------------------------------------------------------------

# Building Recommender 
def critic_recommend(game_name):
    data_index_critic = final_df1_critic[final_df1_critic['title'] == game_name].index[0]
    train_data_reviews_critic[data_index_critic]
        
    new_datapoint_critic = [train_data_reviews_critic[data_index_critic]]
    new_datapoint_critic
    new_vec_critic = nmf.transform(tfidf_vectorizer.transform(new_datapoint_critic))
        
    nn_critic = NearestNeighbors(n_neighbors = 50, metric='cosine', algorithm='brute')
    nn_critic.fit(nmf_tfidf_data)
    result_critic = nn_critic.kneighbors(new_vec_critic)
    result_critic[1][0]
    
    # Generating recommendation list
    recommendation_list_critic=[]
    for r in result_critic[1][0]:
        game_critic = game_id_df2.title[r]
        if game_critic != game_name:
            recommendation_list_critic.append(game_critic)

    return recommendation_list_critic

#------------------------------------------------------------------------------------------------

# Merging sales table
    
#merged_sg = pd.merge(game_info,sales, on='title', how='left')
game_id_df2_w_sales = pd.merge(game_id_df2, sales, on='title', how='left')
game_id_df2_merged = pd.merge(game_id_df2_w_sales, game_info, on='title', how='left')
game_id_df2_merged1 = game_id_df2_merged[['game_id','title','total_sales_USDMM','meta_overview','user_overview']]

#--------------------------------------------------------------------------------------
# Getting hidden gems 

# Getting sum of reviews 
game_info['meta_pos'] = pd.to_numeric(game_info['meta_pos'])
game_info['meta_neg']=pd.to_numeric(game_info['meta_neg'])
game_info['sum_reviews']= game_info.iloc[:,9:12].sum(axis=1)

# Threshold for defining hidden gems
count = 0 
for i in range(len(game_info)):
    if (game_info.sum_reviews[i] <=17):
        count +=1
print(count)

# Hidden gems = games with comments of less than 25(inclusive) and the ones with generally favorable and mixed or average reviews 
# Making a list of hidden gems 
hidden_df_critic = game_info[['title', 'meta_overview','user_overview', 'sum_reviews']]    
hidden_df_critic = hidden_df_critic[hidden_df_critic['sum_reviews'] <=17]
hidden_df_critic = hidden_df_critic[(hidden_df_critic['meta_overview'] == 'Generally favorable reviews') | (hidden_df_critic['meta_overview'] == 'Mixed or average reviews')]

# Total of 103 underexposed games that have generally favorable reviews but lower than 17 comments 

# Checking for how many do we have sales data 
hidden_sales_critic = pd.merge(hidden_df_critic,sales, on='title', how='left')
#*Very few have sales data so not using sales data 

# Merging hidden sales id with the game id
hidden_id_critic = pd.merge(hidden_df_critic,game_id_df2, on='title', how='left')

# Dropping the games that  didn't have game_id
hidden_id_critic = hidden_id_critic.dropna(axis=0)

# Creating a list for hidden gems that can be showcased to people 
hidden_list_critic = hidden_sales_critic['title'].to_list()

hidden_id_list_critic = (hidden_id_critic['game_id']).astype(int).to_list()

# Hidden gems recommender
def hidden_critic_recommend(game_name):
    data_index1_critic = final_df1_critic[final_df1_critic['title'] == game_name].index[0]
    train_data_reviews_critic[data_index1_critic]
        
    new_datapoint1_critic = [train_data_reviews_critic[data_index1_critic]]
    new_vec1_critic = nmf.transform(tfidf_vectorizer.transform(new_datapoint1_critic))
        
    nn1_critic = NearestNeighbors(n_neighbors = 100, metric='cosine', algorithm='brute')
    nn1_critic.fit(nmf_tfidf_data)
    result1_critic = nn1_critic.kneighbors(new_vec1_critic)
    result1_critic[1][0]
    
        
    # Generating recommendation list
    recommendation_list1_critic=[]
    for r in result1_critic[1][0]:
        if r in hidden_id_list_critic:
            game_critic = game_id_df2.title[r]
            if game_critic != game_name:
                recommendation_list1_critic.append(game_critic)
    return recommendation_list1_critic

#------------------------------------------------------------------------------------------------

def listToString(s):  
    str1 = " " 
    return (str1.join(s)) 

game_titles_string = listToString(game_titles)

import random
print('\n********************************************')
print('\n****** Welcome to GameStop RecoBot ******')
print('\n**** The Ultimate Game Recommender ****')
print('\n********************************************')
print("\nHello there! What's your favourite game?\nTry 'explore', if not sure")

message = input('User: ')

def getReply(message):
    if 'Who' in message and 'you' in message:
        reply = "Hi, I'm GameStop RecoBot!" + "\nWould you like to 'explore' games?"
    elif 'explore' in message:
        a = random.sample(game_titles,10)
        reply = "Hmm... Let me see...\nHow about these?\n" + "\n1. " + a[0] + "\n2. "+ a[1] + "\n3. " + a[2] + "\n4. "+ a[3] + "\n5. " + a[4] + "\n6. " + a[5] + "\n7. " + a[6] + "\n8. " + a[7] + "\n9. " + a[8] + "\n10. " + a[9] + "?\n\nI've heard these are pretty interesting!"
    
    elif 'thank' in message:
        reply = "No problem! Adios amigo :)"
    
    elif 'nah' in message or 'nope' in message or 'not' in message or "don't" in message:
        a = random.sample(game_titles,10)
        reply = "Oh okay, no worries!\nHow about these ones?\n" + "\n1. " + a[0] + "\n2. "+ a[1] + "\n3. " + a[2] + "\n4. "+ a[3] + "\n5. " + a[4] + "\n6. " + a[5] + "\n7. " + a[6] + "\n8. " + a[7] + "\n9. " + a[8] + "\n10. " + a[9] 
    
    elif message in game_titles_string:
        game_input = message 
        final_user_recommend = user_recommend(game_input)
        final_hidden_user_recommend = hidden_user_recommend(game_input)
        final_critic_recommend = critic_recommend(game_input)
        final_hidden_critic_recommend = hidden_critic_recommend(game_input)
        reply = "\nGood Choice :D\n"
        print('\n********************************************')
        print("\nThe users recommend: \n1. " + final_user_recommend[0] + "\n2. " +final_user_recommend[1]+ "\n3. " +final_user_recommend[2]+ "\n4. " +final_user_recommend[3]+ "\n5. " +final_user_recommend[4]) 
        print('\n********************************************')
        print("\nHidden gems recommendation from user reviews: \n1. " + final_hidden_user_recommend[0] + "\n2. " +final_hidden_user_recommend[1]+ "\n3. " +final_hidden_user_recommend[2]+ "\n4. " +final_hidden_user_recommend[3]+ "\n5. " +final_hidden_user_recommend[4]) 
        print('\n********************************************')
        print("\nThe critics recommend: \n1. " + final_critic_recommend[1] + "\n2. " +final_critic_recommend[2]+ "\n3. " +final_critic_recommend[3]+ "\n4. " +final_critic_recommend[4]+ "\n5. " +final_critic_recommend[5]) 
        print('\n********************************************')
        print("\nHidden gems recommendation from critic reviews: \n1. " + final_hidden_critic_recommend[0] + "\n2. " +final_hidden_critic_recommend[1]+ "\n3. " +final_hidden_critic_recommend[2]+ "\n4. " +final_hidden_critic_recommend[3]+ "\n5. " +final_hidden_critic_recommend[4]) 
        print('\n********************************************')

    else:
        reply = "Oops... Can't find that game :(\nHow about these?\n" + "\n1. " + a[0] + "\n2. "+ a[1] + "\n3. " + a[2] + "\n4. "+ a[3] + "\n5. " + a[4] 
    print('\nRecBot: ' + reply)

getReply(message)

while ('thank' not in message):
    message = input('User: ')
    getReply(message)


#---------------------------------------------------------------------------------------------------------

'''
# Generating wordcloud for popular vs non popular games 

# Popular games- more than than 25 comments 
# Only games with generally favorable and mixed or average reviews
less_pop_df = pd.merge(user_reviews, hidden_df, on='title', how='left')
less_pop_df = less_pop_df .dropna()
s = less_pop_df['title'].unique()

##Preprocess the text to generate a wordcloud for less poular games

# Lemmatization
less_pop_df['text'] = less_pop_df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
less_pop_df = less_pop_df.reset_index(drop=True)

# Tokenize
summary_words=[]
for i in range(len(less_pop_df)):
    summary_words.append(word_tokenize(less_pop_df['text'][i].lower()))
less_pop_df['summary_clean'] =summary_words


# Filtering away from the comments words that are not neccesary or punctuations.
stopwords = nltk.corpus.stopwords.words('english')
# other common words that can be removed (based on the results of frequency and those that couldn't be taken care of by packages) 
to_remove_words =['game', 'games',"play", "'ll", "una", "40", "ha", "ca","wa", "se", "zelda", "este", "del",  "'s", "n't", "...", '"', "'ve", "``", "'", "'m", "'re", "'d", "si", "30", "9", "te", "60", 'de', 'switch', 'nintendo', 'i', 'que', 'el', 'en', 'es', '10', 'la', 't','un', '2', 'juego', 've', 'm', '3', '1', '5', 'lo', '8', '4', 'd', 'u', '100', 'las','6', '7','co', 'op', 'su']
stopwords.extend(to_remove_words)

punctuation = set(string.punctuation)

# Filter words in stopwords
summary_filt=[]
for i in range(len(less_pop_df)):
    filtered=[]
    for word in less_pop_df['summary_clean'][i]:
        if word not in stopwords:
            filtered.append(word) 
    summary_filt.append(filtered)
less_pop_df['summary_clean']=summary_filt

# Filter punctuation words
word_summary_filt=[]
for i in range(len(less_pop_df)):
    filtered2=[]
    for word in less_pop_df['summary_clean'][i]:
        if word not in punctuation:
            filtered2.append(word) 
    word_summary_filt.append(filtered2)
less_pop_df['summary_clean']=word_summary_filt


# Calculating word frequency in each column
# Ensuring that each word is calculated only once per comment so that we can get accurate results 
less_pop_df["summary_unique"] =less_pop_df["summary_clean"].apply(lambda x: sorted(set(x)))

# Creating a list of all words in the comments of users 
words_summary = less_pop_df["summary_unique"].apply(pd.Series)
words_list = words_summary.stack().tolist()


# Word Cloud for attributes of user's good comments 
unique_string=(" ").join(words_list)
wordcloud = WordCloud(width = 1000, height = 500,  collocations = False, background_color = 'white').generate(unique_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.savefig("your_file_name"+".png", bbox_inches='tight')
plt.show()
plt.close()

# Finding the attrbutes for popular games 

# Making a list of hidden gems 
pop_df = game_info[['title', 'meta_overview','user_overview', 'sum_reviews']]    
pop_df = pop_df[pop_df['sum_reviews'] > 25]
pop_df = pop_df[(pop_df['user_overview'] == 'Generally favorable reviews') | (hidden_df['user_overview'] == 'Mixed or average reviews')]

more_pop_df = pd.merge(user_reviews, pop_df, on='title', how='left')
more_pop_df = more_pop_df .dropna()
st= more_pop_df['title'].unique()

##Preprocess the text to generate a wordcloud for less poular games

# Lemmatization
more_pop_df['text'] = more_pop_df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
more_pop_df = more_pop_df.reset_index(drop=True)

# Tokenize
summary_words1=[]
for i in range(len(more_pop_df)):
    summary_words1.append(word_tokenize(more_pop_df['text'][i].lower()))
more_pop_df['summary_clean'] =summary_words1


# Filtering away from the comments words that are not neccesary or punctuations.

# Filter words in stopwords
summary_filt1=[]
for i in range(len(more_pop_df)):
    filtered1=[]
    for word in more_pop_df['summary_clean'][i]:
        if word not in stopwords:
            filtered1.append(word) 
    summary_filt1.append(filtered1)
more_pop_df['summary_clean']=summary_filt1

# Filter punctuation words
word_summary_filt1=[]
for i in range(len(more_pop_df)):
    filtered2=[]
    for word in more_pop_df['summary_clean'][i]:
        if word not in punctuation:
            filtered2.append(word) 
    word_summary_filt1.append(filtered2)
more_pop_df['summary_clean']=word_summary_filt1


# Calculating word frequency in each column

# Ensuring that each word is calculated only once per comment so that we can get accurate results 
more_pop_df["summary_unique"] =more_pop_df["summary_clean"].apply(lambda x: sorted(set(x)))

# Creating a list of all words in the comments of users 
words_summary1 = more_pop_df["summary_unique"].apply(pd.Series)
words_list1 = words_summary1.stack().tolist()


# Word Cloud for attributes of user's good comments 
unique_string1=(" ").join(words_list1)
wordcloud1 = WordCloud(width = 1000, height = 500,  collocations = False, background_color = 'white').generate(unique_string1)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud1,interpolation="bilinear")
plt.axis("off")
plt.savefig("your_file_name"+".png", bbox_inches='tight')
plt.show()
plt.close()
'''

#----------------------------------------------------------------------------------------------------------