import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import string
import nltk
from potential_talents.tsme_plot import 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from gensim.models import KeyedVectors


# Make sure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_user_keywords(job_title, location):
    # Job title preprocessing
    text = job_title.lower()    # Convert to lowercase
    text = re.sub(r'\d+', '', text)     # remove digits 
    text = text.translate(str.maketrans('','',string.punctuation))   # remove punctuation
    # split the string into tokens
    tokens = nltk.word_tokenize(text)

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Abbreviation expansion
    abbreviation_map = {
        "dev": "developer",
        'chro': 'chief human resources officer',
        'svp': 'senior vice president',
        'gphr': 'global professional in human resources',
        'hris': 'human resources management system',
        'csr': 'corporate social responsibility',
        'sphr': 'strategic and policy-making certification',
        'hr': 'human resources',
        "mgr": "manager",
        "sr": "senior",
        "jr": "junior",
        "eng": "engineer",
        "asst": "assistant",
        "assoc": "associate",
        "admin": "administrator",
        "qa": "quality",
        "vp": "vicepresident",
    }
    # Replace words using abbreviation map dictionary 
    new_tokens = []
    for word in tokens:
        if word in abbreviation_map:
            new_tokens.append(abbreviation_map[word])
        else:
            new_tokens.append(word)

    tokens = new_tokens
    # After all cleaning steps, rejoin tokens into single string
    job_title_cleaned =  ' '.join(tokens)

    # Location Preprocessing 
    location_cleaned = location.lower().strip()
    location_cleaned = re.sub(r'[^\w\s]', '', location_cleaned)

    # Combine processed job_title and location into one string for vectorization
    combined_text = f"{job_title_cleaned} {location_cleaned}"
    return combined_text

def rank_candidates_BoW(keyword, df):
    """ 
    Uses Bag of Words vectorization (BoW - CountVectorizer)
    Combine job_title and location per candidate into a single string
    Compute cosine similarity between candidates and a user provided keyword string
    Rank candidates by fit score and break ties by higher connection counts 
    """
    # Combine job_title and location into one feature for each candidate
    title_location = (df['job_title'] + ' ' + df['location']).astype(str).tolist()

    # Append user keyword to collection, to allow computing similarity between candidates and user query
    title_location.append(keyword)

    # Initialize CountVectorizer (Bag of Words)
    vectorizer = CountVectorizer()

    # fit and transform the collection, learning the vocab from the collection and creating a sparse word count matrix
    vectors = vectorizer.fit_transform(title_location)

    # compute cosine similarity (last vector is the keyword)
    # Between candidate vector which is (vectors[:-1] all except last row)
    # and keyword vector (vectors[-1], last row)
    cosine_sim = cosine_similarity(vectors[:-1], vectors[-1].reshape(1,-1))

    # Add cosine similarity as new column, use flatten() to convert 2D array into 1D array
    df['fit_BoW'] = cosine_sim.flatten()

    # Sort candidates by fit score, then by connection count
    ranked_candidates = df.sort_values(by=['fit_BoW', 'connection'], ascending = [False, False])
    #number of ranked candidates based on the keyword
    print('There are', ranked_candidates[ranked_candidates['fit_BoW']!=0].shape[0],'ranked candidates for the job',keyword)
    return ranked_candidates

def print_candidate_table(df, vector_fit, top_n=15, max_title_len=105):
    """
    Prints a nicely formatted table of the top N ranked candidates with truncated job titles.

    Parameters:
    - df: pandas DataFrame containing at least 'id', 'job_title', 'location', 'connection', and 'fit_SBERT'
    - top_n: number of rows to display (default = 15)
    - max_title_len: maximum length of job title before truncating (default = 105)
    """
    def truncate_string(s, max_len=max_title_len):
        return s if len(s) <= max_len else s[:max_len] + '...'
    # Select subset
    subset = df[['id', 'job_title', 'location', 'connection', vector_fit]].head(top_n).copy()
    # Truncate long job titles
    subset['job_title'] = subset['job_title'].astype(str).apply(lambda x: truncate_string(x))
    # Print as pretty table
    print(tabulate(subset, headers='keys', tablefmt='fancy_grid'))

def rank_candidates_TF_IDF(keyword, df):
    """ 
    Uses TfidfVectorizer, computes cosine similarity between candidates and user keyword
    Appends the TF-IDF based fit score as new column to the passed dataframe
    """
    # combine job title and location for each candidate
    collection = (df['job_title'] + ' ' + df['location']).astype(str).tolist()

    # append the preprocessed keyword to the collection
    collection.append(keyword)

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # fit and transform collection, vectorization
    vectors = tfidf_vectorizer.fit_transform(collection)

    # compute cosine similarity 
    cosine_sim = cosine_similarity(vectors[:-1], vectors[-1].reshape(1,-1))

    # Add TF-IDF cosine similarity score to Dataframe
    df['fit_tfidf'] = cosine_sim.flatten()

    # Sort by TF-IDF fit score and connection count for ranking
    ranked_candidates = df.sort_values(by=['fit_tfidf', 'connection'], ascending=[False, False])
    #number of ranked candidates based on the keyword
    print('There are', ranked_candidates[ranked_candidates['fit_tfidf']!=0].shape[0],'ranked candidates for the job',keyword)
    return ranked_candidates


google_w2v_model_filepath = "../models/GoogleNews-vectors-negative300.bin"
w2v_model = KeyedVectors.load_word2vec_format(google_w2v_model_filepath, binary=True)
def average_word2vec(text, model, embedding_dimension = 300):
    words = text.split()
    # if word exists in model's vocab, collect word vectors
    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])

    # Compute the mean vector
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # else if no words found in vocab, returning zero vector
        return np.zeros(embedding_dimension)
        
def rank_candidates_Word2Vec(keyword, df, model):
    """ 
    Use pretrained Google News Word2ec embeddings
    Compute average Word2Vec vector for each candidate and user keyword
    Compute cosine simlarity and append fit bassed score
    """
    title_location = (df['job_title'] + ' ' + df['location']).astype(str).tolist()

    # Compute Word2Vec average vector for each candidate
    candidate_word2vec = np.array([average_word2vec(string, model) for string in title_location])

    # Compute Word2Vec vector for keyword
    keyword_word2vec = average_word2vec(keyword, model)

    # compute cosine similarity 
    cosine_sim = cosine_similarity(candidate_word2vec, keyword_word2vec.reshape(1,-1))  # reshape to convert keyword vector to 2d array

    # Add the word2vec fit score to the passing dataframe
    df['fit_w2v'] = cosine_sim.flatten()

    ranked_candidates = df.sort_values(by=['fit_w2v', 'connection'], ascending=[False, False])
    print('There are', ranked_candidates[ranked_candidates['fit_w2v']!=0].shape[0],'ranked candidates for the job',keyword)
    return ranked_candidates

if __name__ == "__main__":
    processed_potential_talents = pd.read_csv('../data/interim/processed_potential_talents.csv')

    # user input
    user_job_title = input('Enter job title keyword (example: "Senior Human Resources") :')
    user_location = input('Enter job location (example: "Los Angeles") : ')
    # process user inputs to prepare for cosine similarity
    keyword = preprocess_user_keywords(user_job_title, user_location)

    while True: 
        print("1. Bag of Words")
        print("2. TF-IDF")
        print("3. Word2Vec")
        print("4. GloVe")
        print("5. FastText")
        print("6. BERT")
        print("7. SBERT")

        try:
            method = int(input('Choose number for which vectorization method to perform: '))
        except ValueError:
            print("Invalid input, please enter a number.")
            continue

        if method == 1:
            print("Bag of Words selected.")
            ranked_candidates_df = rank_candidates_BoW(keyword, processed_potential_talents)
            print_candidate_table(ranked_candidates_df, vector_fit="fit_BOW")
        elif method == 2:
            print("Term Frequency - Inverse Document Frequnecy selected.")
            ranked_candidates_df = rank_candidates_TF_IDF(keyword, processed_potential_talents)
            print_candidate_table(ranked_candidates_df, vector_fit="fit_tfidf")
        elif method == 3:
            print("Word2Vec selected.")
            ranked_candidates_df = rank_candidates_TF_IDF(keyword, processed_potential_talents)
            print_candidate_table(ranked_candidates_df, vector_fit="fit_tfidf")