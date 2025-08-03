import pandas as pd
from src.preprocess import clean_reviews
from src.model import train_and_save

def main():
    df = pd.read_csv('data/flipkart_data.csv')
    df_clean = clean_reviews(df)
    train_and_save(df_clean)

if __name__ == '__main__':
    main()
