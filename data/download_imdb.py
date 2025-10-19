import pandas as pd
import numpy as np

# Create a sample IMDB dataset if download fails
def create_sample_imdb_dataset():
    # Create sample reviews and sentiments
    reviews = [
        "This movie was excellent! Great acting and plot.",
        "Terrible waste of time. Poor acting and boring story.",
        "Really enjoyed this film, would watch again!",
        "Disappointing and predictable. Don't bother.",
        "Amazing cinematography and wonderful performances."
    ]
    sentiments = ['positive', 'negative', 'positive', 'negative', 'positive']
    
    # Create DataFrame
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })
    
    # Save to CSV
    df.to_csv('IMDB Dataset.csv', index=False)
    print("Created sample IMDB dataset with 5 entries")

if __name__ == "__main__":
    try:
        create_sample_imdb_dataset()
    except Exception as e:
        print(f"Error: {e}")