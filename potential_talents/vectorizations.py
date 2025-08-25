import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import torch
import string
import nltk
from potential_talents.tsme_plot import plot_embedding_tSNE, print_candidate_table
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.fasttext import load_facebook_vectors    # loading FB's pretrained FastText model
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util 




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
#-----------------------------------------------------------------------------------------------------------------------------------------------

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
    df['embedding'] = list(vectors[:-1].toarray())
    keyword_vector = vectors[-1].toarray().flatten()
    
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

    return ranked_candidates, keyword_vector
#-----------------------------------------------------------------------------------------------------------------------------------------------
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

    df['embedding'] = list(vectors[:-1].toarray())
    keyword_vector = vectors[-1].toarray().flatten()

    # compute cosine similarity 
    cosine_sim = cosine_similarity(vectors[:-1], vectors[-1].reshape(1,-1))

    # Add TF-IDF cosine similarity score to Dataframe
    df['fit_tfidf'] = cosine_sim.flatten()

    # Sort by TF-IDF fit score and connection count for ranking
    ranked_candidates = df.sort_values(by=['fit_tfidf', 'connection'], ascending=[False, False])
    #number of ranked candidates based on the keyword
    print('There are', ranked_candidates[ranked_candidates['fit_tfidf']!=0].shape[0],'ranked candidates for the job',keyword)

    return ranked_candidates, keyword_vector
#-----------------------------------------------------------------------------------------------------------------------------------------------

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
    df = df.copy()
    df['text'] = df['job_title'] + ' ' + df['location']
    title_location = df['text'].astype(str).tolist()

    # Compute Word2Vec average vector for each candidate
    candidate_word2vec = np.array([average_word2vec(string, model) for string in title_location])
    # Compute Word2Vec vector for keyword
    keyword_word2vec = average_word2vec(keyword, model)

    df['embedding'] = list(candidate_word2vec)
    # compute cosine similarity 
    cosine_sim = cosine_similarity(candidate_word2vec, keyword_word2vec.reshape(1,-1))  # reshape to convert keyword vector to 2d array
    # Add the word2vec fit score to the passing dataframe
    df['fit_w2v'] = cosine_sim.flatten()

    ranked_candidates = df.sort_values(by=['fit_w2v', 'connection'], ascending=[False, False])
    print('There are', ranked_candidates[ranked_candidates['fit_w2v']!=0].shape[0],'ranked candidates for the job',keyword)

    return ranked_candidates, keyword_word2vec
#-----------------------------------------------------------------------------------------------------------------------------------------------

# GloVe utilities
# function to convert .txt GloVe file into Word2Vec Format 
def convert_glove_to_word2vec(glove_file_path, word2vec_output_path):
    glove2word2vec(glove_file_path, word2vec_output_path)

# Load pretrained GloVe to return a KeyedVectors object
def load_glove_model(word2vec_output_path):
    model = KeyedVectors.load_word2vec_format(word2vec_output_path, binary=False)
    return model

# Average GloVe embedding for a candidate string
def get_average_glove(text, model, dim=300):
    """ 
    Transforms candidates text or keyword query into fixed size vector to allow for comparison
    between candidates and query in a shared embedding space and use cosine similarity
    """
    words = text.split()
    # collecting vectors for in-vocab words only, otherwise return 0 vector
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0)     # average across all words
    else:
        return np.zeros(dim)

# GloVe function

def rank_candidates_GloVe(keyword, df, model, embedding_dim=300):
    """ 
    Rank candidates in terms of semantic similarity to the user query using dense vector representation.
    Also returns keyword vector for visualization.
    """
    df = df.copy()
    df['text'] = df['job_title'] + ' ' + df['location']
    combined_text = df['text'].astype(str).tolist()

    candidate_vectors = np.array([get_average_glove(text, model, embedding_dim) for text in combined_text])
    keyword_vector = get_average_glove(keyword, model, embedding_dim)

    df['embedding'] = list(candidate_vectors)
    cosine_sim = cosine_similarity(candidate_vectors, keyword_vector.reshape(1, -1))
    df['fit_glove'] = cosine_sim.flatten()

    ranked_candidates = df.sort_values(by=['fit_glove', 'connection'], ascending=[False, False])
    print('There are', ranked_candidates[ranked_candidates['fit_glove'] != 0].shape[0],
          'ranked candidates for the job', keyword)

    return ranked_candidates, keyword_vector
#-----------------------------------------------------------------------------------------------------------------------------------------------
# FASTTEXT
def get_average_fasttext(text_string, model, vector_size=300):
    """ 
    Convert text string into a single fixed-sized vector by averaging word embeddings
    """
    # Return zero vector if text is empty
    if not isinstance(text_string, str) or text_string.strip() == "":
        return np.zeros(vector_size)
    
    words = text_string.lower().split()
    vectors = []
    # retrieve vector from FastText model from each word
    for word in words:
        try:
            vectors.append(model[word])
        except KeyError:
            continue
    # Return zero vector if no valid word vectors were found
    if not vectors:
        return np.zeros(vector_size)
    
    return np.mean(vectors, axis=0)

def rank_candidates_FastText(df, keyword, model, vector_size=300):
    """ 
    Obtain cosine similarity between candidate and keyword vectors using FastText, append fit score to passing dataframe
    """
    df = df.copy()
    df['text'] = df['job_title'] + ' ' + df['location']
    combined_text = df['text'].astype(str).tolist()
    
    # Computing cosine similarity fit score
    # convert list of 300D vectors inot 2D numpy array where each row is candidate vector and column is FastText dimension
    candidate_vectors = np.array([get_average_fasttext(text, model) for text in combined_text])
    keyword_vector = get_average_fasttext(keyword, model, vector_size)

    df['embedding'] = list(candidate_vectors)
    cosine_sim = cosine_similarity(candidate_vectors, keyword_vector.reshape(1,-1))
    df['fit_fasttext'] = cosine_sim.flatten()

    ranked_candidates = df.sort_values(by=['fit_fasttext', 'connection'], ascending=[False, False])
    print('There are', ranked_candidates[ranked_candidates['fit_fasttext']!=0].shape[0],'ranked candidates for the job',keyword)

    return ranked_candidates,keyword_vector
#-----------------------------------------------------------------------------------------------------------------------------------------------
# BERT
def rank_candidates_BERT(df, query_text, model, tokenizer, max_length=64):
    """ 
    Use bert-based-uncased, tokenizes candidate text and query, compuytes the average token embedding, calculates cosine similarity fit scores
    and returns the dataframe with the fit BERT score. 
    Parameters:
    - model: pretrained BERT model
    - tokenizer: matching BERT tokenizer
    - max_length: max token length for truncation
    """
    def get_avg_embedding(text):
        # tokenize the text into a format BERT expects
        input_tokenized = tokenizer(text, return_tensors='pt', truncation=True, padding=True , max_length=max_length)
        # Feed the tokenized inputs into the pretrained BERT model
        with torch.no_grad():   # run in inference mode, disabling gradient tracking
            outputs = model(**input_tokenized)
        # Extract token embeddings, obtaining a matrix of token embeddings
        last_hidden = outputs.last_hidden_state.squeeze(0)
        # returning single 768-dim vector returns NumPy array
        return last_hidden.mean(dim=0).numpy()
    # convert the user query into BERT vector
    query_vec = get_avg_embedding(query_text)

    # combine candidate's datainto a single text string into a 768 dimensional average BERT vector
    combined_texts = (df['job_title'] + ' ' + df['location']).tolist()
    candidate_vectors = np.array([get_avg_embedding(text) for text in combined_texts])

    # Compute cosine similarity between candidate vector and query vector
    # reshape(1,-1) to ensure query is in shape (1,768) and flatten() to 1D array
    df['embedding'] = list(candidate_vectors)
    similarity_scores = cosine_similarity(candidate_vectors, query_vec.reshape(1,-1)).flatten()   
    df['fit_BERT'] = similarity_scores

    ranked_candidates = df.sort_values(by=['fit_BERT', 'connection'], ascending=[False, False])
    print('There are', ranked_candidates[ranked_candidates['fit_BERT']!=0].shape[0],'ranked candidates for the job', query_text)

    return ranked_candidates, query_vec
#-----------------------------------------------------------------------------------------------------------------------------------------------

# SBERT
def rank_candidates_SBERT(df, query, model):
    """ 
    Compute semantic similarity fit scores using a pretrained SBERT Sentence Transformer.
    """
    # Combine candidate text into a unified field
    combined_text = (df['job_title'] + ' ' + df['location']).tolist()

    # Encode Candidate and Keyword texts using model.encode() to convert text into dense sentence embeddings
    # convert_to_tensor=True, returns PyTorch tensors for fast computation
    # normalize_embeddings=True, L2-normalizes each vector
    candidate_embeddings = model.encode(combined_text, convert_to_tensor=True, normalize_embeddings=True)
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

    df['embedding'] = [emb.cpu().numpy() for emb in candidate_embeddings]
    # Compute cosine similarity 
    cosine_scores = util.cos_sim(candidate_embeddings, query_embedding).cpu().numpy().flatten()
    df['fit_SBERT'] = cosine_scores

    ranked_candidates = df.sort_values(by=['fit_SBERT', 'connection'], ascending=[False, False])
    print('There are', ranked_candidates[ranked_candidates['fit_SBERT']!=0].shape[0],'ranked candidates for the job', query)

    return ranked_candidates, query_embedding.cpu().numpy()


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
        print('0. Exit')

        try:
            method = int(input('Choose number for which vectorization method to perform: '))
        except ValueError:
            print("Invalid input, please enter a number.")
            continue

        if method == 1:
            print("Bag of Words selected.")
            ranked_candidates_df, keyboard_vec = rank_candidates_BoW(keyword, processed_potential_talents)
            print_candidate_table(ranked_candidates_df, vector_fit="fit_BoW")
            plot_embedding_tSNE(ranked_candidates_df, fit_col='fit_BoW', keyboard_vector = keyboard_vec, title = 'BoW t-SNE Embedding Visualization')
        elif method == 2:
            print("Term Frequency - Inverse Document Frequnecy selected.")
            ranked_candidates_df, keyboard_vec = rank_candidates_TF_IDF(keyword, processed_potential_talents)
            print_candidate_table(ranked_candidates_df, vector_fit="fit_tfidf")
            plot_embedding_tSNE(ranked_candidates_df, fit_col='fit_tfidf', keyboard_vector = keyboard_vec, title = 'TF-IDF t-SNE Embedding Visualization')
        elif method == 3:
            print("Word2Vec selected.")
            print("Loading Word2Vec Dataset")
            google_w2v_model_filepath = "../models/GoogleNews-vectors-negative300.bin"
            w2v_model = KeyedVectors.load_word2vec_format(google_w2v_model_filepath, binary=True)
            ranked_candidates_df, keyboard_vec = rank_candidates_TF_IDF(keyword, processed_potential_talents)
            print_candidate_table(ranked_candidates_df, vector_fit="fit_w2v")
            plot_embedding_tSNE(ranked_candidates_df, fit_col='fit_w2v', keyboard_vector = keyboard_vec, title = 'Word2Vec t-SNE Embedding Visualization')
        elif method == 4:
            print('GloVe selected.')
            print("Loading pretrained GloVe word vector.")
            glove_filepath = "../models/glove.6B.300d.txt"
            # Convert the GloVe file once
            convert_glove_to_word2vec(glove_filepath, "../models/glove.6B.300d.word2vec.txt")
            # Load the converted file
            glove_model = load_glove_model("../models/glove.6B.300d.word2vec.txt")
            ranked_candidates_df, keyword_vec = rank_candidates_GloVe(keyword, processed_potential_talents, glove_model)
            plot_embedding_tSNE(ranked_candidates_df, fit_col='fit_glove', keyword_vector=keyword_vec, title='GloVe t-SNE Embedding Visualization')
            print_candidate_table(ranked_candidates_df, vector_fit="fit_glove")
        elif method == 5:
            print("FastText selected.")
            # Load Facebook's pretrained FastText model 'cc.en.300.bin'
            fasttext_model = load_facebook_vectors('../models/cc.en.300.bin')
            ranked_candidates_df = rank_candidates_FastText(processed_potential_talents, keyword, fasttext_model)
            plot_embedding_tSNE(ranked_candidates_df, fit_col='fit_fasttext', keyword_vector=keyword_vec, title='FastText t-SNE Embedding Visualization')
            print_candidate_table(ranked_candidates_df, vector_fit="fit_fasttext")
        elif method == 6:
            print("BERT (Bidirectional Encoder Representations from Transformers) selected.")
            # Load BERT-base Model and Tokenizier
            print("Loading Bert-base Model.")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            bert_model.eval() # disable dropout

            ranked_candidates_df, keyword_vec = rank_candidates_BERT(processed_potential_talents, keyword, bert_model, tokenizer)
            plot_embedding_tSNE(ranked_candidates_df, fit_col='fit_BERT', keyword_vector=keyword_vec, title='BERT t-SNE Embedding Visualization')
            print_candidate_table(ranked_candidates_df, vector_fit="fit_BERT")
        elif method == 7:
            print("SBERT (Sentence Bidirectional Encoder Representations from Transformers) selected.")
            # Load SBERT model once 
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loading SBERT model checkpoint all-MiniLM-L6-v2")

            ranked_candidates_df, keyword_vec= rank_candidates_SBERT(processed_potential_talents, keyword, sbert_model)
            plot_embedding_tSNE(ranked_candidates_df, fit_col='fit_SBERT', keyword_vector=keyword_vec, title='SBERT t-SNE Embedding Visualization')
            print_candidate_table(ranked_candidates_df, vector_fit="fit_SBERT")
        elif method == 0:
            print("Exiting.")
            break
        else:
            print("Invalid selection. Please try again.")