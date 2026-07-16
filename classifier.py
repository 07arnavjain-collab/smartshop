import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "review_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "classifier_vectorizer.pkl")

def clean_label(label):
    # Handle string labels like 'OR' (Original -> 0) and 'CG' (Computer Generated -> 1)
    if isinstance(label, str):
        label = label.upper().strip()
        if label == 'OR':
            return 0
        elif label == 'CG':
            return 1
    # Handle numeric or boolean
    try:
        val = int(float(label))
        if val in [0, 1]:
            return val
    except:
        pass
    return 0

def train_classifier(data_path="data/fake_reviews.csv"):
    print(f"Loading fake reviews dataset from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run download_datasets.py first.")
        
    df = pd.read_csv(data_path)
    
    # Check columns
    text_col = 'text_' if 'text_' in df.columns else 'text'
    label_col = 'label'
    
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Required columns (text_ and label) not found. Available: {list(df.columns)}")
        
    # Drop rows with missing text or label
    df = df.dropna(subset=[text_col, label_col])
    
    # Map labels to 0 (Genuine) and 1 (Fake)
    df['target'] = df[label_col].apply(clean_label)
    
    print(f"Dataset size: {len(df)} rows")
    print(f"Class distribution:\n{df['target'].value_counts()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df['target'], test_size=0.2, random_state=42, stratify=df['target']
    )
    
    print("Vectorizing review text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Logistic Regression classifier...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_vec, y_train)
    
    # Evaluate
    predictions = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy on test set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Genuine (OR)", "Fake (CG)"]))
    
    # Save model and vectorizer
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(classifier, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
        
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    return accuracy

def load_classifier():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        classifier = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return classifier, vectorizer

def predict_review(text, classifier=None, vectorizer=None):
    if classifier is None or vectorizer is None:
        classifier, vectorizer = load_classifier()
        if classifier is None or vectorizer is None:
            return "Unknown", 0.5
            
    vec = vectorizer.transform([text])
    pred = classifier.predict(vec)[0]
    prob = classifier.predict_proba(vec)[0][1] # Probability of being Fake (class 1)
    
    label = "Fake" if pred == 1 else "Genuine"
    return label, prob

if __name__ == "__main__":
    train_classifier()
