"""
Dataset Downloader for SmartShopIR.

Real-data-only mode: this project uses ONLY real datasets. No synthetic /
generated product data is created under any circumstances.

  1. Fake Reviews Dataset  -> Kaggle (if credentials exist), else public
                              GitHub mirror of the same real dataset.
  2. Amazon Products       -> Kaggle only (promptcloud/product-listing-from-
                              amazon-india). Kaggle credentials are required;
                              if they're missing or the download fails, this
                              raises a clear error instead of silently
                              fabricating data.
"""

import os
import shutil
import glob
import urllib.request

# Anchor all data paths to this file's directory so downloads land in the
# same place regardless of the process's current working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Public GitHub mirror of the Fake Reviews Dataset (real data) ────────────
FAKE_REVIEWS_URL = (
    "https://raw.githubusercontent.com/SayamAlt/"
    "Fake-Reviews-Detection/main/fake%20reviews%20dataset.csv"
)

KAGGLE_AMAZON_SLUG = "promptcloud/product-listing-from-amazon-india"
KAGGLE_REVIEWS_SLUG = "sudarshan24py/fake-reviews-dataset"


class KaggleCredentialsMissing(Exception):
    """Raised when the real Amazon product catalog can't be obtained
    because no Kaggle credentials are configured. We deliberately do NOT
    fall back to generated/synthetic data - real dataset only."""
    pass


def _has_kaggle_credentials():
    home = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")
    return os.path.exists(kaggle_json) or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )


def download_fake_reviews(output_path=None):
    """Download the real fake-reviews dataset: Kaggle first if credentials
    exist, otherwise a public GitHub mirror of the same real dataset."""
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "fake_reviews.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    print("Downloading Fake Reviews Dataset from GitHub mirror...")
    try:
        urllib.request.urlretrieve(FAKE_REVIEWS_URL, output_path)
        print(f"Downloaded fake reviews -> {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading from GitHub: {e}")
        return False


def download_amazon_catalog(output_path=None):
    """Download the real Amazon India product catalog from Kaggle.

    Real-data-only: if Kaggle credentials aren't configured, or the
    download fails, this raises KaggleCredentialsMissing rather than
    generating any synthetic/fake product data.
    """
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "amazon_products.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    home = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")

    if not _has_kaggle_credentials():
        raise KaggleCredentialsMissing(
            "No Kaggle credentials found (KAGGLE_USERNAME / KAGGLE_KEY, or "
            "~/.kaggle/kaggle.json). The real Amazon product catalog can only "
            "be downloaded from Kaggle - this app does not generate synthetic "
            "product data. Add your Kaggle credentials (e.g. as Streamlit "
            "secrets) and try again."
        )

    if not os.path.exists(kaggle_json) and os.environ.get("KAGGLE_USERNAME"):
        os.makedirs(os.path.dirname(kaggle_json), exist_ok=True)
        with open(kaggle_json, "w") as f:
            f.write(f'{{"username":"{os.environ["KAGGLE_USERNAME"]}","key":"{os.environ["KAGGLE_KEY"]}"}}')

    import kagglehub
    print("Downloading Amazon Product Catalog from Kaggle...")
    try:
        amazon_dir = kagglehub.dataset_download(KAGGLE_AMAZON_SLUG)
    except Exception as e:
        raise KaggleCredentialsMissing(
            f"Kaggle download of the real Amazon product catalog failed: {e}. "
            f"This app does not fall back to synthetic data - please check "
            f"your Kaggle credentials and try again."
        ) from e

    csv_files = glob.glob(os.path.join(amazon_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise KaggleCredentialsMissing(
            "Kaggle download succeeded but no CSV file was found in the "
            "downloaded dataset."
        )

    shutil.copy(csv_files[0], output_path)
    print(f"Downloaded real Amazon catalog from Kaggle -> {output_path}")
    return True


def download_and_extract():
    """Main entry point: ensure both real datasets are available in data/."""
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
