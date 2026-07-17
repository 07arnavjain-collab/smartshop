import os
import pickle
import hashlib
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
import re

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def stable_hash(value: str) -> int:
    """Deterministic string hash that stays the same across process restarts.

    Python's built-in hash() is randomized per-process for security reasons,
    so using it for anything that needs to be reproducible between runs
    (e.g. picking a consistent set of reviews for a product) will silently
    give different results every time the app restarts.
    """
    return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)


# Anchor all data/model paths to this file's directory rather than using bare
# relative paths. Relative paths like "data/x.pkl" only resolve correctly if
# the process's current working directory happens to be the project root -
# which isn't guaranteed (e.g. a subprocess, a different launch command, or
# some hosting environments can start the app from elsewhere). When the path
# doesn't resolve, os.path.exists() just returns False and the code silently
# falls back to "no data" instead of erroring - which is exactly what caused
# reviews to silently disappear. Anchoring to __file__ makes this robust.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_products.pkl")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "models", "faiss_index.bin")
BM25_INDEX_PATH = os.path.join(BASE_DIR, "models", "bm25_index.pkl")

class SmartShopSearch:
    def __init__(self, sample_size=5000):
        self.sample_size = sample_size
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = None
        self.df = None
        self.faiss_index = None
        self.bm25 = None
        self.tokenized_corpus = None
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Remove special characters and lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def clean_price(self, price_str):
        if pd.isna(price_str) or not isinstance(price_str, str):
            return 0.0
        # Extract digits, commas, and periods
        nums = re.findall(r'[\d\.]+', price_str.replace(',', ''))
        if nums:
            try:
                return float(nums[0])
            except ValueError:
                return 0.0
        return 0.0

    def clean_image_url(self, url_str):
        if pd.isna(url_str) or not isinstance(url_str, str):
            # Return placeholder
            return "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=500&auto=format&fit=crop"
        # Split by pipe or comma if multiple urls exist
        urls = re.split(r'[\|\,]', url_str)
        cleaned_url = urls[0].strip()
        if not cleaned_url.startswith("http"):
            return "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=500&auto=format&fit=crop"
        return cleaned_url

    def clean_category(self, cat_str):
        if pd.isna(cat_str) or not isinstance(cat_str, str):
            return "General"
        # Often categories are formatted like: "Clothing >> Men's >> Shirts"
        cats = [c.strip() for c in cat_str.split(">>")]
        return cats[0] if cats else "General"

    def load_and_preprocess_catalog(self, raw_path=None):
        if raw_path is None:
            raw_path = os.path.join(BASE_DIR, "data", "amazon_products.csv")
        print(f"Preprocessing raw catalog from {raw_path}...")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw catalog file not found at {raw_path}")
            
        df = pd.read_csv(raw_path)
        
        # Select and rename columns for standard schema
        # Expected: Product Title, Product Description, Category, Price, MRP, Image URLs, Uniq Id
        col_mapping = {
            'Product Title': 'title',
            'product_title': 'title',
            'Product Description': 'description',
            'product_description': 'description',
            'Category': 'category',
            'category': 'category',
            'Price': 'price',
            'price': 'price',
            'MRP': 'mrp',
            'mrp': 'mrp',
            'Image URLs': 'image_url',
            'image_url': 'image_url',
            'Uniq Id': 'uniq_id',
            'uniq_id': 'uniq_id',
            'Brand': 'brand',
            'brand': 'brand'
        }
        
        df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
        
        # Fill missing required columns
        for col in ['title', 'description', 'category', 'price', 'mrp', 'image_url', 'uniq_id', 'brand']:
            if col not in df.columns:
                df[col] = ""
                
        # Sample for speed and memory efficiency
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
            
        # Perform cleaning
        df['title'] = df['title'].fillna("Unnamed Product")
        df['description'] = df['description'].fillna("No description available.")
        df['category_clean'] = df['category'].apply(self.clean_category)
        df['price_clean'] = df['price'].apply(self.clean_price)
        df['mrp_clean'] = df['mrp'].apply(self.clean_price)
        df['image_clean'] = df['image_url'].apply(self.clean_image_url)
        df['brand'] = df['brand'].fillna("Generic")
        
        # Ensure ratings are clean
        if 'ratings' in df.columns:
            df['rating'] = pd.to_numeric(df['ratings'].str.extract(r'([\d\.]+)')[0], errors='coerce').fillna(4.0)
        else:
            # Generate deterministic ratings based on uniq_id hash
            df['rating'] = df['uniq_id'].apply(lambda x: 3.0 + (stable_hash(str(x)) % 21) / 10.0)
            
        self.df = df
        
        # Save processed data
        os.makedirs("data", exist_ok=True)
        self.df.to_pickle(PROCESSED_DATA_PATH)
        print(f"Processed {len(df)} products and saved to {PROCESSED_DATA_PATH}")

    def build_indexes(self):
        if self.df is None:
            if os.path.exists(PROCESSED_DATA_PATH):
                self.df = pd.read_pickle(PROCESSED_DATA_PATH)
            else:
                raise ValueError("Dataframe not loaded. Call load_and_preprocess_catalog first.")
                
        print("Initializing SentenceTransformer model...")
        self.model = SentenceTransformer(self.model_name)
        
        # 1. Build BM25 Lexical Index
        print("Building BM25 index...")
        stop_words = set(stopwords.words('english'))
        
        corpus = (self.df['title'].astype(str) + " " + self.df['description'].astype(str)).tolist()
        self.tokenized_corpus = []
        for doc in corpus:
            cleaned = self.clean_text(doc)
            tokens = [w for w in cleaned.split() if w not in stop_words]
            self.tokenized_corpus.append(tokens)
            
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Save BM25 index
        os.makedirs("models", exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump((self.bm25, self.tokenized_corpus), f)
        print(f"BM25 index saved to {BM25_INDEX_PATH}")
        
        # 2. Build FAISS Semantic Index
        print("Generating dense embeddings for semantic search...")
        # Embed Title + Description for richer matching vocabulary
        search_texts = (
            self.df['title'].astype(str) + ". " +
            self.df['category_clean'].astype(str) + ". " +
            self.df['description'].astype(str).str[:200]
        ).tolist()
        embeddings = self.model.encode(search_texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity via Inner Product
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        
        # Save FAISS Index
        faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
        print(f"FAISS index saved to {FAISS_INDEX_PATH}")

    def load_indexes(self):
        if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(BM25_INDEX_PATH):
            return False
            
        print("Loading precomputed data and indexes...")
        self.df = pd.read_pickle(PROCESSED_DATA_PATH)
        
        self.model = SentenceTransformer(self.model_name)
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25, self.tokenized_corpus = pickle.load(f)
            
        return True

    def search(self, query, top_k=20, lexical_weight=0.3, category_filter=None,
               min_price=None, max_price=None, min_rating=None, min_score=0.52):
        if self.df is None:
            self.load_indexes()

        # Clean query
        cleaned_query = self.clean_text(query)
        query_tokens = [w for w in cleaned_query.split()
                        if w not in set(stopwords.words('english'))]

        # 1. Lexical Scoring (BM25)
        bm25_scores = (
            np.array(self.bm25.get_scores(query_tokens))
            if query_tokens else np.zeros(len(self.df))
        )
        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # 2. Semantic Scoring (FAISS)
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search all items and remap back to dataframe order
        semantic_scores_raw, indices = self.faiss_index.search(query_embedding, len(self.df))
        semantic_scores_raw = semantic_scores_raw[0]
        indices = indices[0]

        mapped_semantic_scores = np.zeros(len(self.df))
        for score, idx in zip(semantic_scores_raw, indices):
            mapped_semantic_scores[idx] = max(score, 0.0)   # clamp negatives

        # Cosine similarities from normalized FAISS are already in [0,1] range
        # (dot product of two unit vectors = cos similarity, capped at 1)

        # 3. Hybrid Combination
        hybrid_scores = (
            lexical_weight * bm25_scores +
            (1 - lexical_weight) * mapped_semantic_scores
        )

        # Build results dataframe with scores
        results = self.df.copy()
        results['search_score'] = hybrid_scores
        results['lexical_score'] = bm25_scores
        results['semantic_score'] = mapped_semantic_scores

        # Apply minimum relevance threshold — suppress truly irrelevant results
        results = results[results['search_score'] >= min_score]

        # Apply user filters
        if category_filter and category_filter != "All":
            results = results[results['category_clean'] == category_filter]
        if min_price is not None:
            results = results[results['price_clean'] >= min_price]
        if max_price is not None:
            results = results[results['price_clean'] <= max_price]
        if min_rating is not None:
            results = results[results['rating'] >= min_rating]

        # Sort and return top K
        results = results.sort_values(by='search_score', ascending=False).head(top_k)
        return results

    def get_recommendations(self, product_idx, top_n=5):
        if self.df is None:
            self.load_indexes()
            
        # Get query embedding for the source product from index
        # We can extract the embedding of this item by query or direct FAISS lookup
        # For content recommendation, we can rebuild queries using the product's title + category
        prod = self.df.iloc[product_idx]
        text = str(prod['title']) + " in " + str(prod['category_clean'])
        
        embedding = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        
        # Search FAISS (request top_n + 1 to exclude the product itself)
        scores, indices = self.faiss_index.search(embedding, top_n + 1)
        
        rec_indices = [idx for idx in indices[0] if idx != product_idx][:top_n]
        return self.df.iloc[rec_indices]

# Review Associator to dynamically link Fake Reviews dataset to catalog
class ReviewAssociator:
    def __init__(self, fake_reviews_path=None):
        self.fake_reviews_path = fake_reviews_path or os.path.join(BASE_DIR, "data", "fake_reviews.csv")
        self.reviews_df = None
        
    def load_reviews(self):
        if not os.path.exists(self.fake_reviews_path):
            return False
        df = pd.read_csv(self.fake_reviews_path)
        # Standardize columns
        self.reviews_df = df.rename(columns={'text_': 'text', 'text': 'text'})
        return True
        
    def get_reviews_for_product(self, product_id, category, num_reviews=6):
        if self.reviews_df is None:
            if not self.load_reviews():
                return []
                
        # Find reviews matching category if possible, else general reviews
        # Fake reviews category names look like 'Home_and_Kitchen_5', 'Electronics_5',
        # 'Sports_and_Outdoors_5' - the trailing "_<digit>" is a star-rating subset
        # marker from the source dataset and must be stripped before comparing.
        # Clean Flipkart/Amazon category: 'Home & Kitchen', 'Electronics', etc.
        cat_clean = category.lower().replace(" ", "").replace("&", "and")

        reviews_cat_clean = (
            self.reviews_df['category'].astype(str)
            .str.lower()
            .str.replace(r"_\d+$", "", regex=True)  # strip trailing star-rating suffix
            .str.replace("_", "")
            .str.replace(" ", "")
        )
        matching = self.reviews_df[reviews_cat_clean == cat_clean]
        
        if len(matching) < num_reviews:
            # Fallback to general reviews
            matching = self.reviews_df
            
        # Deterministic sampling based on product_id hash so product reviews are
        # consistent across app restarts (stable_hash, not the built-in hash()).
        hash_val = stable_hash(str(product_id))
        sample_indices = []
        for i in range(num_reviews):
            idx = (hash_val + i * 997) % len(matching)
            sample_indices.append(idx)
            
        sampled = matching.iloc[sample_indices].copy()
        
        # Map labels to 0 (Genuine) and 1 (Fake) for standard representation
        if 'label' in sampled.columns:
            def clean(lbl):
                if isinstance(lbl, str):
                    return 1 if lbl.strip().upper() == 'CG' else 0
                return int(lbl)
            sampled['is_fake'] = sampled['label'].apply(clean)
        else:
            sampled['is_fake'] = 0
            
        return sampled.to_dict('records')
