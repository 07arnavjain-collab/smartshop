"""
Dataset Downloader for SmartShopIR.

Strategy:
  1. Fake Reviews Dataset  -> Download from Kaggle (primary) or public GitHub raw URL (fallback).
  2. Amazon Products       -> Download from Kaggle (credentials required).
"""

import os
import shutil
import glob
import urllib.request

# Anchor all data paths to this file's directory so downloads land in the
# same place regardless of the process's current working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Public GitHub mirror of the Fake Reviews Dataset ─────────────────────────
FAKE_REVIEWS_URL = (
    "https://raw.githubusercontent.com/SayamAlt/"
    "Fake-Reviews-Detection/main/fake%20reviews%20dataset.csv"
)

# ── Kaggle dataset slugs ─────────────────────────────────────────────────────
KAGGLE_AMAZON_SLUG = "promptcloud/product-listing-from-amazon-india"
KAGGLE_REVIEWS_SLUG = "sudarshan24py/fake-reviews-dataset"


def download_fake_reviews(output_path=None):
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "fake_reviews.csv")
    """Download fake reviews dataset from Kaggle or public GitHub mirror."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try Kaggle first if credentials available
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        try:
            import kagglehub
            reviews_dir = kagglehub.dataset_download(KAGGLE_REVIEWS_SLUG)
            csv_files = glob.glob(os.path.join(reviews_dir, "**", "*.csv"), recursive=True)
            if csv_files:
                shutil.copy(csv_files[0], output_path)
                print(f"Downloaded fake reviews from Kaggle -> {output_path}")
                return True
        except Exception as e:
            print(f"Kaggle download failed ({e}), falling back to GitHub mirror...")

    # Fall back to public GitHub mirror
    print(f"Downloading Fake Reviews Dataset from GitHub mirror...")
    try:
        urllib.request.urlretrieve(FAKE_REVIEWS_URL, output_path)
        print(f"Downloaded fake reviews -> {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading from GitHub: {e}")
        return False


def download_amazon_catalog(output_path=None):
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "amazon_products.csv")
    """Download Amazon catalog from Kaggle. Credentials are required."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check for Kaggle credentials
    home = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")

    has_creds = os.path.exists(kaggle_json) or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )

    if not has_creds:
        raise RuntimeError(
            "Kaggle API credentials are required to download the Amazon Products dataset. "
            "Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables or place a "
            "kaggle.json file in ~/.kaggle/."
        )

    # Write kaggle.json from env if needed
    if not os.path.exists(kaggle_json) and os.environ.get("KAGGLE_USERNAME"):
        os.makedirs(os.path.dirname(kaggle_json), exist_ok=True)
        with open(kaggle_json, "w") as f:
            f.write(f'{{"username":"{os.environ["KAGGLE_USERNAME"]}","key":"{os.environ["KAGGLE_KEY"]}"}}')

    try:
        import kagglehub
        print("Downloading Amazon Product Catalog from Kaggle...")
        amazon_dir = kagglehub.dataset_download(KAGGLE_AMAZON_SLUG)
        csv_files = glob.glob(os.path.join(amazon_dir, "**", "*.csv"), recursive=True)
        if csv_files:
            shutil.copy(csv_files[0], output_path)
            print(f"Downloaded Amazon catalog from Kaggle -> {output_path}")
            return True
        else:
            raise FileNotFoundError("No CSV files found in the downloaded Kaggle dataset.")
    except Exception as e:
        raise RuntimeError(f"Failed to download Amazon catalog from Kaggle: {e}")


def download_and_extract():
    """Main entry point: ensure both datasets are available in data/."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(DATA_DIR, "amazon_products.csv")):
        download_amazon_catalog()
    else:
        print("Amazon products dataset already exists locally.")

    if not os.path.exists(os.path.join(DATA_DIR, "fake_reviews.csv")):
        download_fake_reviews()
    else:
        print("Fake reviews dataset already exists locally.")


if __name__ == "__main__":
    download_and_extract()
