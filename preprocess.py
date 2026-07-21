import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Change into the project directory so every relative path used anywhere in
# this pipeline (here and in the modules it imports) resolves consistently,
# regardless of what directory the process was actually launched from. This
# is what silently broke review lookups before: the app process and this
# script's subprocess could end up with different working directories,
# so "data/fake_reviews.csv" pointed at two different places.
os.chdir(BASE_DIR)

from download_datasets import download_and_extract
from classifier import train_classifier
from search_engine import SmartShopSearch

def main():
    print("=== STARTING SMARTSHOPIR SETUP & PREPROCESSING ===")
    
    # 1. Download datasets if needed
    if not os.path.exists("data/amazon_products.csv") or not os.path.exists("data/fake_reviews.csv"):
        print("Datasets not found locally. Starting downloader...")
        download_and_extract()
    else:
        print("Found Amazon products and Fake Reviews datasets locally in 'data/'.")

    # Double check that files exist before proceeding
    if not os.path.exists("data/amazon_products.csv"):
        print("ERROR: data/amazon_products.csv is missing. Cannot proceed.")
        return
    if not os.path.exists("data/fake_reviews.csv"):
        print("ERROR: data/fake_reviews.csv is missing. Cannot proceed.")
        return

    # 2. Train Review Classifier
    print("\n--- Training Fake Review Detection Classifier ---")
    try:
        if not os.path.exists("models/review_classifier.pkl") or not os.path.exists("models/classifier_vectorizer.pkl"):
            train_classifier("data/fake_reviews.csv")
        else:
            print("Trained review classifier and vectorizer already exist. Skipping training.")
    except Exception as e:
        print(f"Error during classifier training: {e}")
        return

    # 3. Preprocess Product Catalog and Build Search Indexes
    print("\n--- Building Product Search Catalog and Vector Indexes ---")
    try:
        # Create Search object (we limit to 5000 products for quick indexing/fast search)
        search_engine = SmartShopSearch(sample_size=5000)
        search_engine.load_and_preprocess_catalog("data/amazon_products.csv")
        search_engine.build_indexes()
        print("Catalog preprocessed and indexes successfully generated.")
    except Exception as e:
        print(f"Error during indexing: {e}")
        return

    print("\n=== SETUP & PREPROCESSING COMPLETE ===")
    print("You can now run the Streamlit app: streamlit run app.py")

if __name__ == "__main__":
    main()
