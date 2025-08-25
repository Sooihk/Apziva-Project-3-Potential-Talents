from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from potential_talents.tsme_plot import plot_embedding_tSNE, print_candidate_table


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
    print('There are', ranked_candidates[ranked_candidates['fit_SBERT']!=0].shape[0],'ranked candidates for the job',keyword)

    return ranked_candidates, query_embedding.cpu().numpy()
#-----------------------------------------------------------------------------------------------------------------------------------------------
# SBERT Rerank

def rerank_candidates_with_sbert(df, query_embedding, model, starred_id, alpha=0.6):
    """ 
    Re-rank candidates based on SBERT cosine similarity, with bias toward a user-starred candidate. 

    Parameters:
    - df: Dataframe
    - query: original job query
    - model: SentenceTransformer, preloaded SBERT model
    - starred_id: int, ID of the starred candidiate
    - alpha: float, weight for the starred candidate (0 < alpha < 1), higher values have more influence from starred profile,
             how much influence the user-starred candidiate has on the reranked results
    """
      # Prepare text
    df = df.copy()
    df["text"] = df["job_title"] + " " + df["location"]

    # Encode query and candidates into vectors
    candidate_embeddings = model.encode(df["text"].tolist(), normalize_embeddings=True)

     # Get starred candidate embedding
    try:
        starred_text = df[df["id"] == starred_id]["text"].values[0]
    except IndexError:
        raise ValueError(f"Candidate ID {starred_id} not found in DataFrame.")
    # encode embedding into semnatic vector using SBERT
    starred_embedding = model.encode(starred_text, normalize_embeddings=True)

    # Comptue cosine similarity before blending
    cos_before = util.cos_sim(starred_embedding, query_embedding).item()
    # Weighted combination of query and starred embeddings
    combined_embedding = (1 - alpha) * query_embedding + alpha * starred_embedding
    # Compute cosine similarity after blending
    cos_after = util.cos_sim(starred_embedding, combined_embedding).item()

    print(f"Cosine similarity before reranking: {cos_before:.4f}")
    print(f"Cosine similarity after reranking: {cos_after:.4f}")
    if cos_after > cos_before:
        print("Blended query is now closer to the starred candidate (semantic gain).")
    else:
        print("Blended query did not move closer to the starred candidate.")

    # Rerank based on new embedding
    df["fit_SBERT_RE"] = util.cos_sim(candidate_embeddings, combined_embedding).squeeze().numpy()
    df_sorted = df.sort_values(by=["fit_SBERT_RE", 'connection'], ascending=False)

    return df_sorted, combined_embedding

if __name__ == "__main__":
    processed_potential_talents = pd.read_csv('../data/interim/processed_potential_talents.csv')

    # user input
    user_job_title = input('Enter job title keyword (example: "Senior Human Resources") :')
    user_location = input('Enter job location (example: "Los Angeles") : ')
    # process user inputs to prepare for cosine similarity
    keyword = preprocess_user_keywords(user_job_title, user_location)

    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    # start ranking
    rank_candidates_df, keyword_vec = rank_candidates_SBERT(processed_potential_talents)
    plot_embedding_tSNE(rank_candidates_df, fit_col='fit_SBERT', keyword_vector=keyword_vec, title = 'SBERT t-SNE Embedding Visualization.')
    print_candidate_table(rank_candidates_df, vector_fit='fit_SBERT')

    while True:
        try:
            user_id = int(input("\nRerank the candidate table by entering a candidate's ID number you favor (or type 0 to exit): "))
            if user_id == 0:
                print("Exiting reranking.")
                break


            if user_id not in rank_candidates_df['id'].values:
                print(f"Candidate ID {user_id} not found in the candidate list. Please try again.")
                continue


            rank_candidates_df, keyword_vec = rerank_candidates_with_sbert(rank_candidates_df, keyword_vec, sbert_model, user_id)
            plot_embedding_tSNE(rank_candidates_df, fit_col='fit_SBERT_RE', keyword_vector=keyword_vec, title='SBERT Reranked t-SNE Visualization')
            print_candidate_table(rank_candidates_df, vector_fit='fit_SBERT_RE')


        except ValueError:
            print("Invalid input. Please enter a numeric ID.")

