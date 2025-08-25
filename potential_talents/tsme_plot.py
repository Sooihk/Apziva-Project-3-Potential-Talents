from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import numpy as np
from tabulate import tabulate

def plot_embedding_tSNE(df, fit_col, keyword_vector, keyword_label='KEYWORD', label_col='job_title', top_n=50, title='t-SNE Embedding Visualization'):
    """
    General-purpose t-SNE plot function for any vectorization method including keyword point.

    Parameters:
    - df: DataFrame with 'embedding' and fit scores
    - fit_col: Name of the fit score column (e.g., 'fit_sbert')
    - keyword_vector: numpy array representing the vector of the keyword
    - keyword_label: label to display for the keyword point
    - label_col: Column name to label candidate points (e.g., 'job_title')
    - top_n: Number of top candidates to display
    - title: Plot title
    """
    # Select top candidates
    top_candidates = df.sort_values(by=fit_col, ascending=False).head(top_n)

    if 'embedding' not in top_candidates.columns:
        raise ValueError("Expected column 'embedding' containing candidate vectors.")

    vectors = np.stack(top_candidates['embedding'].values)
    fit_scores = top_candidates[fit_col].tolist()
    labels = [label if len(label) <= 50 else label[:50] + '...' for label in top_candidates[label_col].tolist()]

    # Group into fit score categories
    quantiles = np.quantile(fit_scores, [0.33, 0.66])
    fit_groups = [
        'Low Fit' if score <= quantiles[0]
        else 'Medium Fit' if score <= quantiles[1]
        else 'High Fit'
        for score in fit_scores
    ]

    # Append keyword vector and metadata
    vectors = np.vstack([vectors, keyword_vector])
    labels.append(keyword_label)
    fit_groups.append('Keyword')

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_result = tsne.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(14, 10))
    palette = {'Low Fit': 'red', 'Medium Fit': 'orange', 'High Fit': 'green', 'Keyword': 'blue'}
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=fit_groups, palette=palette, s=80, edgecolor='k')

    texts = []
    for i, label in enumerate(labels):
        texts.append(plt.text(tsne_result[i, 0], tsne_result[i, 1], label, fontsize=9))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))
    plt.title(title, fontweight='bold')
    plt.legend(title='Fit Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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