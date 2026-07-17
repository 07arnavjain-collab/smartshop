"""
Dataset Downloader for SmartShopIR.

Strategy:
  1. Fake Reviews Dataset  -> Download from public GitHub raw URL (no credentials needed).
  2. Amazon Products       -> If Kaggle credentials exist, pull from Kaggle.
                             Otherwise, generate a high-quality synthetic catalog
                             modeled on the PromptCloud/Amazon-India schema.
"""

import os
import shutil
import glob
import re
import random
import csv
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

# ── Kaggle dataset slug (used when credentials ARE available) ─────────────────
KAGGLE_AMAZON_SLUG = "promptcloud/product-listing-from-amazon-india"
KAGGLE_REVIEWS_SLUG = "sudarshan24py/fake-reviews-dataset"

# ── Synthetic product catalog parameters ─────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books",
    "Sports & Outdoors", "Beauty & Personal Care", "Toys & Games",
    "Health & Wellness", "Automotive", "Jewellery"
]

BRANDS = {
    "Electronics": ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple", "Samsung", "Sony", "OnePlus", "Xiaomi", "JBL", "Logitech", "boAt", "Realme", "Motorola"],
    "Clothing": ["Levi's", "Nike", "Puma", "H&M", "Zara", "Adidas", "Allen Solly"],
    "Home & Kitchen": ["Prestige", "Pigeon", "Borosil", "Milton", "Cello", "Solimo"],
    "Books": ["Penguin", "HarperCollins", "Westland", "Rupa Publications", "S.Chand"],
    "Sports & Outdoors": ["Nivia", "Cosco", "Yonex", "Decathlon", "Vector X"],
    "Beauty & Personal Care": ["Lakme", "L'Oreal", "Mamaearth", "Biotique", "Dove"],
    "Toys & Games": ["Funskool", "Hasbro", "Mattel", "Lego", "Fisher-Price"],
    "Health & Wellness": ["Himalaya", "Dabur", "Healthkart", "Patanjali", "Dr. Morepen"],
    "Automotive": ["Bosch", "3M", "Michelin", "Castrol", "Havoline"],
    "Jewellery": ["Tanishq", "Malabar Gold", "Kalyan Jewellers", "PC Jeweller", "Giva"]
}

PRODUCT_TEMPLATES = {
    "Electronics": [
        # Laptops
        "{brand} {laptop_line} Laptop - Intel Core {cpu} {ram}GB RAM {storage}GB SSD {gpu_tag}",
        "{brand} {laptop_line} {screen_size}\" Laptop {ram}GB RAM {storage}GB SSD - {color}",
        "{brand} Gaming Laptop {ram}GB RAM {storage}GB SSD {gpu} GPU",
        "{brand} {adj} Thin & Light Laptop Intel {cpu} {ram}GB {storage}GB SSD",
        "{brand} {laptop_line} 2-in-1 Touchscreen Laptop {ram}GB RAM {storage}GB",
        # Smartphones
        "{brand} {phone_model} 5G Smartphone {ram}GB RAM {storage}GB | {mp}MP Camera",
        "{brand} {phone_model} {ram}GB/{storage}GB {mp}MP Triple Camera Smartphone - {color}",
        "{brand} {adj} Android Smartphone {screen_size}\" AMOLED {ram}GB RAM",
        # Tablets
        "{brand} {adj} Android Tablet {screen_size}\" {ram}GB RAM {storage}GB Storage",
        "{brand} {adj} Tablet {screen_size}\" FHD Display {ram}GB RAM Wi-Fi",
        # Audio
        "{brand} {adj} Wireless Bluetooth Earbuds with {feature} and {hrs}hrs Battery",
        "{brand} {adj} Over-Ear Headphones with Noise Cancellation - {color}",
        "{brand} Portable Bluetooth Speaker with {feature} and {hrs}hrs Playback",
        # Monitors & Peripherals
        "{brand} {screen}\" Full HD IPS Monitor with {feature}",
        "{brand} {adj} Mechanical Gaming Keyboard - {color} Backlit",
        "{brand} {adj} Wireless Mouse {dpi} DPI - {color}",
        # Accessories
        "{brand} {adj} USB-C Fast Charger {watts}W with {feature}",
        "{brand} {adj} Smart Watch with {feature} and {hrs}hrs Battery",
        "{brand} {mp}MP Digital Camera with {feature} Lens",
        "{brand} {screen}\" 4K Smart LED TV with {feature}",
    ],
    "Clothing": [
        "{brand} Men's {adj} {fabric} {type} - {color}",
        "{brand} Women's {adj} {fabric} {dress_type} for {occasion}",
        "{brand} Unisex {adj} {jacket_type} Jacket - {color}",
        "{brand} Men's {fabric} Slim Fit Jeans - {color}",
        "{brand} Women's {adj} Casual {type} - {color}",
    ],
    "Home & Kitchen": [
        "{brand} {capacity}L {adj} Pressure Cooker with {feature}",
        "{brand} {adj} Non-Stick {cookware} Set of {count} - {color}",
        "{brand} {watts}W {adj} Mixer Grinder with {count} Jars",
        "{brand} {capacity}L Insulated Water Bottle - {color}",
        "{brand} {adj} Stainless Steel Dinner Set - {count} Pieces",
    ],
    "Books": [
        "{title} by {author} - Paperback Edition",
        "{title}: {subtitle} - {author}",
        "The Complete Guide to {topic} - {author}",
        "{title} (Illustrated Edition) - {author}",
        "{adj} {topic}: A Comprehensive Study by {author}",
    ],
    "Sports & Outdoors": [
        "{brand} {adj} Cricket Bat - {wood} Willow Grade {grade}",
        "{brand} {adj} Badminton Racket with Full Cover",
        "{brand} {adj} Football - Size {size} with {feature}",
        "{brand} {adj} Yoga Mat {thickness}mm - {color}",
        "{brand} {adj} Running Shoes for Men - {color}",
    ],
    "Beauty & Personal Care": [
        "{brand} {adj} Moisturizing Face Cream with {ingredient} - {ml}ml",
        "{brand} {adj} Hair Oil with {ingredient} for Nourishment - {ml}ml",
        "{brand} {adj} Sunscreen SPF {spf} with {ingredient} - {ml}ml",
        "{brand} {adj} Vitamin C Face Serum - {ml}ml",
        "{brand} {adj} Body Lotion with {ingredient} for Soft Skin - {ml}ml",
    ],
    "Toys & Games": [
        "{brand} {adj} Building Blocks Set - {count} Pieces",
        "{brand} {adj} Remote Control Car with {feature}",
        "{brand} {adj} Educational Board Game for Kids",
        "{brand} {adj} Plush Stuffed Animal Toy - {size}",
        "{brand} {adj} Drawing and Art Set for Kids - {count} Colors",
    ],
    "Health & Wellness": [
        "{brand} {adj} Whey Protein {flavor} Flavor - {weight}kg",
        "{brand} {adj} Multivitamin Tablets - Pack of {count}",
        "{brand} {adj} Digital Blood Pressure Monitor",
        "{brand} {adj} Ayurvedic {herbal} Capsules - {count} Count",
        "{brand} {adj} Protein Bar {flavor} - Pack of {count}",
    ],
    "Automotive": [
        "{brand} {adj} Car Dash Camera with {feature}",
        "{brand} {adj} Engine Oil {grade} - {capacity}L",
        "{brand} {adj} Car Air Purifier with HEPA Filter",
        "{brand} {adj} Tyre Inflator with Digital Gauge",
        "{brand} {adj} Car Seat Cover Set - {color}",
    ],
    "Jewellery": [
        "{brand} {adj} Sterling Silver {design} Pendant Necklace",
        "{brand} {adj} Rose Gold Plated {design} Earrings",
        "{brand} {adj} 22K Gold Plated {design} Bracelet",
        "{brand} {adj} Diamond Studded {design} Ring",
        "{brand} {adj} Pearl {design} Jewellery Set",
    ],
}

DESCRIPTIONS = {
    "Electronics": "Experience cutting-edge performance with this {brand} device. Engineered for power users, students, and professionals alike. Features the latest processor technology, long battery life, fast connectivity, and a premium build that stands up to daily use. Whether you need it for work, gaming, creative tasks, or entertainment — this product delivers.",
    "Clothing": "Crafted from premium {fabric}, this {brand} garment offers superior comfort and style. Perfect for everyday wear and special occasions alike. Machine washable and designed to retain its shape and colour wash after wash.",
    "Home & Kitchen": "This {brand} kitchen essential is designed to make your cooking experience effortless. Made from high-quality materials, it is built to last and easy to clean. The ergonomic design ensures comfortable handling every day.",
    "Books": "A must-read for anyone interested in this subject. Written in clear, engaging prose that makes complex topics accessible to all readers. Includes detailed illustrations, examples, and exercises.",
    "Sports & Outdoors": "Designed for performance and durability, this {brand} sports product is ideal for both amateurs and professionals. Engineered to withstand rigorous use while maintaining peak performance.",
    "Beauty & Personal Care": "This {brand} beauty product is formulated with the finest natural ingredients to nourish and protect your skin. Dermatologically tested and free from harmful chemicals. Suitable for all skin types.",
    "Toys & Games": "This {brand} toy is designed to spark creativity and imagination in young minds. Made from safe, non-toxic materials and tested to meet international safety standards. Ideal for children aged 3 and above.",
    "Health & Wellness": "Support your health journey with this premium {brand} wellness product. Formulated by health experts using scientifically-backed ingredients to help you achieve your fitness and wellness goals.",
    "Automotive": "Keep your vehicle in top condition with this {brand} automotive product. Engineered to meet OEM specifications and tested for reliability in all driving conditions.",
    "Jewellery": "Adorn yourself with this exquisite {brand} jewellery piece, crafted with attention to detail and finished to perfection. Perfect as a gift or a personal treat.",
}

FILL = {
    "adj": ["Premium", "Ultra", "Pro", "Advanced", "Deluxe", "Classic", "Smart", "Eco"],
    "feature": ["Noise Cancellation", "Fast Charging", "Water Resistance", "Active Cooling",
                 "HD Display", "Anti-Scratch Coating", "360° Sound", "AI Enhancement",
                 "Thunderbolt 4", "Wi-Fi 6", "USB-C", "Backlit Keyboard"],
    "hrs": list(range(8, 40, 4)),
    "screen": [24, 27, 32, 40, 43, 50, 55, 65],
    "screen_size": [13.3, 14, 15.6, 16, 17.3],
    "mp": [12, 24, 48, 64, 108],
    "watts": [18, 30, 45, 65, 100],
    # Laptop-specific fills
    "laptop_line": ["Inspiron", "Pavilion", "IdeaPad", "VivoBook", "Aspire",
                    "ProBook", "ThinkPad", "ZenBook", "TUF Gaming", "Nitro 5",
                    "Latitude", "Envy", "Spectre", "Swift", "Chromebook"],
    "cpu": ["i3-1215U", "i5-1235U", "i5-12500H", "i7-1255U", "i7-12700H",
            "i9-12900H", "Ryzen 5 5500U", "Ryzen 7 5800H", "Ryzen 9 6900HX",
            "Core Ultra 5", "Core Ultra 7", "M2", "M3 Pro"],
    "ram": [4, 8, 16, 32, 64],
    "storage": [128, 256, 512, 1000, 2000],
    "gpu": ["NVIDIA RTX 3050", "NVIDIA RTX 3060", "NVIDIA RTX 4060",
             "NVIDIA RTX 4070", "AMD Radeon RX 6600M", "Intel Arc A370M"],
    "gpu_tag": ["", "- Integrated Graphics", "- Discrete GPU", "- GeForce GPU"],
    # Smartphone fills
    "phone_model": ["Galaxy S23", "Galaxy A54", "Nord CE3", "Redmi Note 13",
                    "POCO X6 Pro", "Pixel 8a", "Moto G84", "Find X7",
                    "Reno 11", "Narzo 70 Pro", "Y200", "iQOO Neo 9"],
    "dpi": [800, 1200, 1600, 2400, 3200, 6400],
    # Clothing
    "fabric": ["Cotton", "Polyester", "Linen", "Denim", "Silk", "Wool", "Fleece"],
    "type": ["T-Shirt", "Shirt", "Kurta", "Polo", "Hoodie"],
    "dress_type": ["Kurti", "Dress", "Top", "Saree", "Lehenga"],
    "jacket_type": ["Bomber", "Denim", "Windbreaker", "Parka", "Fleece"],
    "occasion": ["Casual", "Formal", "Party", "Festive", "Office"],
    "color": ["Black", "White", "Navy Blue", "Olive Green", "Maroon", "Beige", "Grey",
              "Silver", "Space Grey", "Midnight Blue", "Starlight"],
    "capacity": [1, 1.5, 2, 3, 5, 10],
    "cookware": ["Kadai", "Fry Pan", "Tawa", "Wok", "Saucepan"],
    "count": list(range(3, 16)),
    "title": ["The Art of Thinking", "Digital Horizons", "Mindful Leadership",
              "Data Secrets", "The Wellness Code", "Beyond Limits", "Silent Signals"],
    "subtitle": ["A Modern Approach", "Principles and Practices", "The Definitive Guide",
                 "Strategies for Success", "Mastering the Fundamentals"],
    "author": ["Rajesh Kumar", "Priya Sharma", "Arjun Nair", "Neha Gupta", "Dr. Vikram Mehta"],
    "topic": ["Python", "Machine Learning", "Indian History", "Philosophy", "Economics"],
    "wood": ["English", "Kashmir"],
    "grade": [1, 2, 3],
    "size": [3, 4, 5],
    "thickness": [4, 6, 8],
    "ingredient": ["Aloe Vera", "Vitamin E", "Turmeric", "Neem", "Rose Water", "Argan Oil"],
    "ml": [50, 100, 150, 200, 250, 500],
    "spf": [15, 30, 50],
    "size_str": ["Small", "Medium", "Large", "30cm"],
    "flavor": ["Chocolate", "Vanilla", "Strawberry", "Mango", "Unflavored"],
    "weight": [1, 2, 5],
    "herbal": ["Ashwagandha", "Triphala", "Brahmi", "Giloy", "Shilajit"],
    "grade_oil": ["5W-30", "10W-40", "15W-40"],
    "design": ["Floral", "Heart", "Geometric", "Traditional", "Modern"],
}


def _fill_template(template, brand, category):
    result = template.replace("{brand}", brand)
    for key, values in FILL.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, str(random.choice(values)))
    return result


def _make_price():
    base = random.choice([99, 199, 299, 499, 699, 999, 1299, 1499, 1999,
                          2499, 2999, 3499, 4999, 5999, 7999, 9999, 12999, 14999, 19999])
    discount = random.uniform(0.05, 0.40)
    discounted = round(base * (1 - discount), -1)
    return base, int(discounted)


def _generate_rows_for_category(cat, count, placeholder_images):
    """Generate `count` synthetic product rows for a single category."""
    brands_in_cat = BRANDS[cat]
    templates = PRODUCT_TEMPLATES[cat]
    desc_template = DESCRIPTIONS[cat]
    rows = []

    for _ in range(count):
        brand = random.choice(brands_in_cat)
        tmpl = random.choice(templates)
        title = _fill_template(tmpl, brand, cat)
        desc = desc_template.replace("{brand}", brand).replace("{fabric}", random.choice(FILL["fabric"]))
        mrp, price = _make_price()
        rating = round(random.uniform(3.0, 5.0), 1)
        uid = f"amz_{cat[:3].upper()}_{random.randint(100000, 999999)}"

        rows.append({
            "Uniq Id": uid,
            "Product Title": title,
            "Product Description": desc,
            "Category": cat,
            "Brand": brand,
            "MRP": f"₹{mrp:,}",
            "Price": f"₹{price:,}",
            "Ratings": str(rating),
            "Image URLs": placeholder_images[cat],
        })
    return rows


PLACEHOLDER_IMAGES = {
    "Electronics": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=500&auto=format&fit=crop",
    "Clothing": "https://images.unsplash.com/photo-1489987707025-afc232f7ea0f?w=500&auto=format&fit=crop",
    "Home & Kitchen": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=500&auto=format&fit=crop",
    "Books": "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=500&auto=format&fit=crop",
    "Sports & Outdoors": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=500&auto=format&fit=crop",
    "Beauty & Personal Care": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=500&auto=format&fit=crop",
    "Toys & Games": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500&auto=format&fit=crop",
    "Health & Wellness": "https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?w=500&auto=format&fit=crop",
    "Automotive": "https://images.unsplash.com/photo-1605559424843-9e4c228bf1c2?w=500&auto=format&fit=crop",
    "Jewellery": "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=500&auto=format&fit=crop",
}


def generate_synthetic_amazon_catalog(output_path=None, n=5000):
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "amazon_products.csv")
    """Generate a synthetic Amazon India-style product catalog."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "Uniq Id", "Product Title", "Product Description",
        "Category", "Brand",
        "MRP", "Price",
        "Ratings", "Image URLs"
    ]

    rows = []
    per_category = n // len(CATEGORIES)

    for cat in CATEGORIES:
        rows.extend(_generate_rows_for_category(cat, per_category, PLACEHOLDER_IMAGES))

    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} synthetic Amazon products -> {output_path}")


def ensure_category_coverage(output_path=None, min_per_category=150):
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "amazon_products.csv")
    """
    Top up any category that is missing or under-represented in the catalog
    (common with real scraped data, which may have zero rows for a whole
    category like Electronics) by injecting synthetic products for just
    that category. Real rows are always kept; only the gap is filled.
    """
    import pandas as pd

    if not os.path.exists(output_path):
        return

    df = pd.read_csv(output_path)
    if "Category" not in df.columns:
        # Can't reliably tell what's covered without a category column;
        # leave the file untouched rather than guess.
        return

    # Only look at the top-level category (before any ">>" nesting) so real
    # scraped taxonomies like "Electronics >> Computers >> Laptops" still count.
    top_level = df["Category"].astype(str).str.split(">>").str[0].str.strip()

    added_rows = []
    for cat in CATEGORIES:
        existing_count = (top_level.str.lower() == cat.lower()).sum()
        if existing_count < min_per_category:
            needed = min_per_category - existing_count
            print(f"  '{cat}' has only {existing_count} real products — "
                  f"topping up with {needed} synthetic listings.")
            added_rows.extend(_generate_rows_for_category(cat, needed, PLACEHOLDER_IMAGES))

    if not added_rows:
        print("Every category already has enough products — no top-up needed.")
        return

    added_df = pd.DataFrame(added_rows)
    # Align columns: keep whatever real-data columns exist, fill the rest as blank
    for col in df.columns:
        if col not in added_df.columns:
            added_df[col] = ""
    added_df = added_df[df.columns]

    combined = pd.concat([df, added_df], ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"Catalog now has {len(combined)} total products "
          f"({len(added_rows)} synthetic top-up rows added).")


def download_fake_reviews(output_path=None):
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "fake_reviews.csv")
    """Download fake reviews dataset from public GitHub mirror."""
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
    """Download Amazon catalog from Kaggle if credentials exist, else generate synthetic."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try Kaggle if credentials available
    home = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")

    has_creds = os.path.exists(kaggle_json) or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )

    if has_creds:
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
                print("Checking category coverage in the real dataset...")
                ensure_category_coverage(output_path)
                return True
        except Exception as e:
            print(f"Kaggle download failed ({e}), generating synthetic catalog instead...")

    # Fallback: generate synthetic catalog
    print("Generating synthetic Amazon product catalog (5,000 products)...")
    generate_synthetic_amazon_catalog(output_path, n=5000)
    return True


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
