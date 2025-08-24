from pathlib import Path
# basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_talents():
    # Load csv dataset
    potential_talents_filepath = '../data/raw/potential-talents.xlsx'
    potential_talents = pd.read_excel(potential_talents_filepath)
    potential_talents.drop('fit',axis=1, inplace=True)
    return potential_talents


if __name__ == "__main__":
    df = load_talents()
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("Preview:")
    print(df.info())
    df.head()