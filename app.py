import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from search_engine import SmartShopSearch, ReviewAssociator
from classifier import predict_review, load_classifier
import subprocess
import time

# Set page configuration
st.set_page_config(
    page_title="SmartShopIR - AI E-Commerce Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark Theme and Glassmorphism
CUSTOM_CSS = """
<style>
    /* Main container background */
    .stApp {
        background-color: #0A0A12;
        color: #E6E6FA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #A88BEB 0%, #F1A7F1 50%, #FF9DE2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #8C8CA3;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Cards for product results */
    .product-card {
        background: rgba(18, 18, 37, 0.6);
        border: 1px solid rgba(108, 92, 231, 0.15);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }
    
    .product-card:hover {
        transform: translateY(-4px);
        border-color: rgba(108, 92, 231, 0.5);
        box-shadow: 0 12px 40px 0 rgba(108, 92, 231, 0.2);
    }
    
    /* Review box styling */
    .review-box {
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 12px;
        border: 1px solid;
    }
    
    .genuine-review {
        background: rgba(46, 213, 115, 0.05);
        border-color: rgba(46, 213, 115, 0.2);
    }
    
    .fake-review {
        background: rgba(255, 71, 87, 0.05);
        border-color: rgba(255, 71, 87, 0.2);
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6C5CE7 0%, #8F75FF 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #8F75FF 0%, #A88BEB 100%);
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.5);
    }
    
    /* Badges */
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-category {
        background: rgba(108, 92, 231, 0.2);
        color: #A88BEB;
    }
    
    .badge-genuine {
        background: rgba(46, 213, 115, 0.2);
        color: #2ED573;
    }
    
    .badge-fake {
        background: rgba(255, 71, 87, 0.2);
        color: #FF4757;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper to verify indices
def system_initialized():
    return (
        os.path.exists("data/processed_products.pkl") and
        os.path.exists("models/faiss_index.bin") and
        os.path.exists("models/bm25_index.pkl") and
        os.path.exists("models/review_classifier.pkl")
    )

# Header
st.markdown("<h1 class='main-header'>SmartShopIR</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Hybrid Search & Fraud Review Analytics Engine</p>", unsafe_allow_html=True)

# Check initialization state
if not system_initialized():
    st.warning("⚠️ SmartShopIR system components are not initialized.")
    st.info("The raw Kaggle datasets need to be downloaded, processed, and search indexes built. This will take a moment.")
    
    # Prompt for credentials if they don't exist
    home = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")
    
    has_creds = os.path.exists(kaggle_json) or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    
    if not has_creds:
        st.subheader("🔑 Configure Kaggle API Credentials")
        st.write("To download datasets directly from Kaggle, please input your username and API key (found in Kaggle -> Account Settings -> Create New Token).")
        col1, col2 = st.columns(2)
        with col1:
            username_input = st.text_input("Kaggle Username", key="setup_username")
        with col2:
            key_input = st.text_input("Kaggle API Key", type="password", key="setup_key")
            
        if st.button("Save Credentials & Initialize"):
            if username_input and key_input:
                os.makedirs(os.path.dirname(kaggle_json), exist_ok=True)
                with open(kaggle_json, "w") as f:
                    f.write(f'{{"username":"{username_input}","key":"{key_input}"}}')
                os.environ["KAGGLE_USERNAME"] = username_input
                os.environ["KAGGLE_KEY"] = key_input
                st.success("Credentials saved!")
                st.rerun()
            else:
                st.error("Please fill in both username and key.")
    else:
        st.write("✅ Kaggle API Credentials detected.")
        if st.button("Initialize System Now"):
            with st.spinner("Setting up system... This downloads ~60MB of data and trains models... Please wait."):
                # Run preprocessor
                try:
                    result = subprocess.run(["python", "preprocess.py"], capture_output=True, text=True, check=True)
                    st.success("System initialized successfully!")
                    st.text(result.stdout)
                    time.sleep(2)
                    st.rerun()
                except subprocess.CalledProcessError as e:
                    st.error(f"Failed to initialize system: {e}")
                    st.text(e.stdout)
                    st.text(e.stderr)
    st.stop()

# Initialize Backend Instances
@st.cache_resource
def load_search_instance():
    search = SmartShopSearch()
    search.load_indexes()
    return search

@st.cache_resource
def load_review_associator():
    associator = ReviewAssociator()
    associator.load_reviews()
    return associator

@st.cache_resource
def load_classifier_model():
    return load_classifier()

search_engine = load_search_instance()
review_associator = load_review_associator()
classifier, vectorizer = load_classifier_model()

# Sidebar Filters & Configuration
st.sidebar.markdown("## ⚙️ Search Configuration")

# Weight distribution
lexical_weight = st.sidebar.slider(
    "Search Weighting",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Higher values favor keyword matching (BM25). Lower values favor semantic meaning matching (Dense Embeddings)."
)

st.sidebar.markdown("### 📊 Catalog Filters")
# Extract categories list
categories = ["All"] + sorted(list(search_engine.df['category_clean'].unique()))
category_filter = st.sidebar.selectbox("Filter Category", categories)

price_min, price_max = float(search_engine.df['price_clean'].min()), float(search_engine.df['price_clean'].max())
if price_max <= price_min:
    price_max = price_min + 1000
    
price_range = st.sidebar.slider(
    "Price Range (₹)",
    min_value=0.0,
    max_value=10000.0,  # Cap at reasonable max for visualization
    value=(0.0, 5000.0),
    step=50.0
)

rating_min = st.sidebar.slider(
    "Minimum Rating",
    min_value=1.0,
    max_value=5.0,
    value=1.0,
    step=0.5
)

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Smart Search",
    "🛡️ Review Authenticator",
    "⚖️ Compare Products",
    "📊 Market Analytics"
])

# ----------------- Tab 1: Smart Search -----------------
with tab1:
    st.subheader("🔍 E-Commerce Search")
    
    query = st.text_input("Enter keywords or describe what product you are looking for:", placeholder="e.g., sterling silver heart pendant necklace")
    
    if query:
        # Perform hybrid search
        results = search_engine.search(
            query=query,
            top_k=20,
            lexical_weight=lexical_weight,
            category_filter=category_filter,
            min_price=price_range[0],
            max_price=price_range[1],
            min_rating=rating_min
        )
        
        if results.empty:
            st.info("No products found matching your search and filter criteria.")
        else:
            st.write(f"Showing top {len(results)} matches for '{query}':")
            
            for idx, row in results.iterrows():
                # Get reviews associated to calculate authentic reviews ratio
                reviews = review_associator.get_reviews_for_product(row['uniq_id'], row['category_clean'])
                
                # Check reviews status using trained classifier model
                genuine_cnt = 0
                for r in reviews:
                    lbl, prob = predict_review(r['text'], classifier, vectorizer)
                    if lbl == "Genuine":
                        genuine_cnt += 1
                        
                authenticity_rate = (genuine_cnt / len(reviews)) * 100 if reviews else 100.0
                
                # Render HTML card
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    st.image(row['image_clean'], use_container_width=True)
                    
                with col2:
                    st.markdown(f"### {row['title']}")
                    st.markdown(
                        f"<span class='badge badge-category'>{row['category_clean']}</span> "
                        f"&nbsp; Brand: **{row['brand']}**",
                        unsafe_allow_html=True
                    )
                    
                    st.write(row['description'][:300] + ("..." if len(str(row['description'])) > 300 else ""))
                    
                    price_display = f"₹{row['price_clean']:.2f}" if row['price_clean'] > 0 else "Price not available"
                    mrp_display = f"₹{row['mrp_clean']:.2f}" if row['mrp_clean'] > 0 else ""
                    
                    col_p, col_r, col_a, col_d = st.columns([2, 2, 2.5, 2.5])
                    with col_p:
                        if mrp_display:
                            st.markdown(f"**Price:** {price_display} <span style='text-decoration: line-through; color: #8C8CA3; font-size: 0.8rem;'>{mrp_display}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**Price:** {price_display}")
                    with col_r:
                        st.markdown(f"⭐ **{row['rating']:.1f}** / 5.0")
                    with col_a:
                        auth_color = "green" if authenticity_rate >= 70 else "red" if authenticity_rate < 40 else "orange"
                        st.markdown(f"🛡️ Trust Rate: <span style='color:{auth_color}; font-weight:bold;'>{authenticity_rate:.0f}%</span>", unsafe_allow_html=True)
                    with col_d:
                        if st.button("View Product Details", key=f"btn_{row['uniq_id']}"):
                            st.session_state['selected_product'] = row.to_dict()
                            st.session_state['selected_product_reviews'] = reviews
                            st.session_state['selected_product_index'] = search_engine.df[search_engine.df['uniq_id'] == row['uniq_id']].index[0]
                            st.rerun()
                
                st.markdown("<hr style='border: 0.5px solid rgba(255,255,255,0.05)'>", unsafe_allow_html=True)

    # Detailed View Overlay/Display
    if 'selected_product' in st.session_state:
        p = st.session_state['selected_product']
        revs = st.session_state['selected_product_reviews']
        p_idx = st.session_state['selected_product_index']
        
        st.markdown("<br><hr style='border: 1px solid rgba(108, 92, 231, 0.4)'><br>", unsafe_allow_html=True)
        st.subheader(f"🔍 Product Detailed Review: {p['title']}")
        
        det_col1, det_col2 = st.columns([2, 3])
        with det_col1:
            st.image(p['image_clean'], use_container_width=True)
            st.markdown(f"**Category:** {p['category']}")
            st.markdown(f"**Brand:** {p['brand']}")
            st.markdown(f"**Unique ID:** `{p['uniq_id']}`")
            if p['price_clean'] > 0:
                st.markdown(f"### Price: ₹{p['price_clean']:.2f}")
            else:
                st.markdown("### Price: N/A")
                
        with det_col2:
            st.markdown("#### Description")
            st.write(p['description'])
            
            st.markdown("#### Product Authenticity Analytics")
            
            # Recalculate review labels
            labeled_revs = []
            fake_count = 0
            for r in revs:
                pred_label, score = predict_review(r['text'], classifier, vectorizer)
                labeled_revs.append({'text': r['text'], 'label': pred_label, 'score': score})
                if pred_label == "Fake":
                    fake_count += 1
                    
            trust_percent = (1 - (fake_count / len(revs))) * 100 if revs else 100
            
            # Simple progress bar
            st.progress(trust_percent / 100.0)
            st.markdown(f"Of the parsed reviews, **{trust_percent:.1f}%** are classified as **Genuine** (Organic) and **{100-trust_percent:.1f}%** as **Fake** (Computer Generated).")
            
            # Show reviews list
            st.markdown("#### Customer Reviews Auditing")
            for idx, r in enumerate(labeled_revs):
                box_class = "genuine-review" if r['label'] == "Genuine" else "fake-review"
                badge_class = "badge-genuine" if r['label'] == "Genuine" else "badge-fake"
                
                st.markdown(
                    f"<div class='review-box {box_class}'>"
                    f"  <span class='badge {badge_class}'>{r['label'].upper()}</span> "
                    f"  <span style='font-size: 0.8rem; color: #8C8CA3;'> (Fake Prob: {r['score']:.1%})</span>"
                    f"  <p style='margin-top: 8px;'>\"{r['text']}\"</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
        # Recommendations section
        st.markdown("### 🏷️ Similar Products You Might Like")
        recs = search_engine.get_recommendations(p_idx, top_n=4)
        rec_cols = st.columns(4)
        for i, (r_idx, rec_row) in enumerate(recs.iterrows()):
            with rec_cols[i]:
                st.image(rec_row['image_clean'], use_container_width=True)
                st.markdown(f"**{rec_row['title'][:40]}...**")
                st.markdown(f"₹{rec_row['price_clean']:.2f} | ⭐{rec_row['rating']:.1f}")
                if st.button("View Product", key=f"rec_{rec_row['uniq_id']}"):
                    rec_reviews = review_associator.get_reviews_for_product(rec_row['uniq_id'], rec_row['category_clean'])
                    st.session_state['selected_product'] = rec_row.to_dict()
                    st.session_state['selected_product_reviews'] = rec_reviews
                    st.session_state['selected_product_index'] = r_idx
                    st.rerun()

# ----------------- Tab 2: Review Authenticator -----------------
with tab2:
    st.subheader("🛡️ Real-Time Review Authenticity Checker")
    st.write("Write or paste a product review below to analyze if it is likely genuine or computer-generated (fake).")
    
    test_review_text = st.text_area(
        "Paste Customer Review Text:",
        height=150,
        placeholder="Type something like: 'This product is absolutely amazing! Highly recommend buying it right now.' OR 'Automated review test product review good quality fast delivery.'"
    )
    
    if st.button("Run Authenticity Audit"):
        if test_review_text.strip():
            label, score = predict_review(test_review_text, classifier, vectorizer)
            
            st.markdown("### Analysis Results")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if label == "Genuine":
                    st.success("✅ **GENUINE REVIEW**")
                    st.write("The model classifies this review as **Organic/Genuine**.")
                else:
                    st.error("🚨 **POTENTIALLY FAKE REVIEW**")
                    st.write("The model classifies this review as **Computer Generated or Deceptive/Fake**.")
                    
                st.metric(label="Calculated Fake Probability", value=f"{score:.2%}")
                
            with col_res2:
                # Plotly Gauge Chart for visual representation
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fakery Score (%)", 'font': {'size': 20, 'color': '#E6E6FA'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#E6E6FA"},
                        'bar': {'color': "#6C5CE7"},
                        'bgcolor': "rgba(18, 18, 37, 0.6)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(108, 92, 231, 0.3)",
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(46, 213, 115, 0.2)'},
                            {'range': [40, 70], 'color': 'rgba(255, 165, 2, 0.2)'},
                            {'range': [70, 100], 'color': 'rgba(255, 71, 87, 0.2)'}
                        ]
                    }
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#E6E6FA"},
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please input some text before running the audit.")

# ----------------- Tab 3: Compare Products -----------------
with tab3:
    st.subheader("⚖️ Compare Products Side-by-Side")
    st.write("Select products from the catalog to compare pricing, ratings, description details, and review authenticity levels.")
    
    # Selection
    product_options = search_engine.df[['uniq_id', 'title']].to_dict('records')
    options_dict = {f"{item['title'][:60]}... (ID: {item['uniq_id'][:8]})": item['uniq_id'] for item in product_options}
    
    selected_options = st.multiselect(
        "Select up to 4 products to compare:",
        options=list(options_dict.keys()),
        max_selections=4
    )
    
    if selected_options:
        ids_to_compare = [options_dict[opt] for opt in selected_options]
        compare_df = search_engine.df[search_engine.df['uniq_id'].isin(ids_to_compare)].copy()
        
        # Build comparison grid
        compare_records = []
        for idx, row in compare_df.iterrows():
            # Get reviews statistics
            reviews = review_associator.get_reviews_for_product(row['uniq_id'], row['category_clean'])
            genuine_cnt = 0
            for r in reviews:
                lbl, _ = predict_review(r['text'], classifier, vectorizer)
                if lbl == "Genuine":
                    genuine_cnt += 1
            auth_rate = (genuine_cnt / len(reviews)) * 100 if reviews else 100.0
            
            compare_records.append({
                "Image": f'<img src="{row["image_clean"]}" width="100">',
                "Title": row['title'],
                "Brand": row['brand'],
                "Category": row['category_clean'],
                "Price": f"₹{row['price_clean']:.2f}" if row['price_clean'] > 0 else "N/A",
                "Original Price (MRP)": f"₹{row['mrp_clean']:.2f}" if row['mrp_clean'] > 0 else "N/A",
                "Rating": f"⭐ {row['rating']:.1f} / 5.0",
                "Review Trust Index": f"🛡️ {auth_rate:.0f}% Genuine"
            })
            
        comp_display = pd.DataFrame(compare_records).T
        comp_display.columns = [f"Product {i+1}" for i in range(len(selected_options))]
        
        # Render table with HTML allowed for image URLs
        st.write(comp_display.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.info("Please select products using the dropdown above.")

# ----------------- Tab 4: Market Analytics -----------------
with tab4:
    st.subheader("📊 Catalog & Review Market Analytics")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### Product Categories Pricing Breakdown")
        # Category average price
        cat_stats = search_engine.df.groupby('category_clean')['price_clean'].mean().reset_index().sort_values(by='price_clean', ascending=False)
        fig_price = px.bar(
            cat_stats,
            x='price_clean',
            y='category_clean',
            orientation='h',
            labels={'price_clean': 'Average Price (₹)', 'category_clean': 'Category'},
            color='price_clean',
            color_continuous_scale='Purples'
        )
        fig_price.update_layout(
            paper_bgcolor='rgba(18, 18, 37, 0.6)',
            plot_bgcolor='rgba(18, 18, 37, 0.6)',
            font={'color': "#E6E6FA"},
            margin=dict(l=20, r=20, t=30, b=20),
            height=350
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
    with col_chart2:
        st.markdown("#### Overall Star Rating Distribution")
        fig_rating = px.histogram(
            search_engine.df,
            x='rating',
            nbins=10,
            labels={'rating': 'Product Rating'},
            color_discrete_sequence=['#8F75FF']
        )
        fig_rating.update_layout(
            paper_bgcolor='rgba(18, 18, 37, 0.6)',
            plot_bgcolor='rgba(18, 18, 37, 0.6)',
            font={'color': "#E6E6FA"},
            margin=dict(l=20, r=20, t=30, b=20),
            height=350
        )
        st.plotly_chart(fig_rating, use_container_width=True)
        
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.markdown("#### Review Classification Distribution")
        # Load sample reviews labels to show general distribution
        if review_associator.reviews_df is not None:
            lbl_counts = review_associator.reviews_df['label'].value_counts().reset_index()
            lbl_counts.columns = ['Review Label', 'Count']
            
            # Map strings to human labels if needed
            lbl_counts['Review Label'] = lbl_counts['Review Label'].apply(
                lambda x: "Fake (Computer Generated)" if str(x).upper() == 'CG' or x == 1 else "Genuine (Organic)"
            )
            
            fig_pie = px.pie(
                lbl_counts,
                values='Count',
                names='Review Label',
                color_discrete_sequence=['#FF4757', '#2ED573'],
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(18, 18, 37, 0.6)',
                plot_bgcolor='rgba(18, 18, 37, 0.6)',
                font={'color': "#E6E6FA"},
                margin=dict(l=20, r=20, t=30, b=20),
                height=350
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Reviews dataset not loaded for pie chart.")
            
    with col_chart4:
        st.markdown("#### Authenticity Trust Rate by Product Category")
        # Sample some products across categories and calculate average trust rating
        cats_to_plot = search_engine.df['category_clean'].unique()[:8]
        cats_trust = []
        
        for cat in cats_to_plot:
            products_in_cat = search_engine.df[search_engine.df['category_clean'] == cat].head(10)
            trust_scores = []
            for _, p_row in products_in_cat.iterrows():
                revs = review_associator.get_reviews_for_product(p_row['uniq_id'], cat, num_reviews=4)
                genuine_cnt = sum(1 for r in revs if predict_review(r['text'], classifier, vectorizer)[0] == "Genuine")
                trust_scores.append((genuine_cnt / len(revs)) * 100 if revs else 100.0)
            cats_trust.append({
                "Category": cat,
                "Average Trust Rate (%)": np.mean(trust_scores) if trust_scores else 100.0
            })
            
        trust_df = pd.DataFrame(cats_trust)
        fig_trust = px.bar(
            trust_df,
            x='Category',
            y='Average Trust Rate (%)',
            range_y=[0, 100],
            color='Average Trust Rate (%)',
            color_continuous_scale='RdYlGn'
        )
        fig_trust.update_layout(
            paper_bgcolor='rgba(18, 18, 37, 0.6)',
            plot_bgcolor='rgba(18, 18, 37, 0.6)',
            font={'color': "#E6E6FA"},
            margin=dict(l=20, r=20, t=30, b=20),
            height=350
        )
        st.plotly_chart(fig_trust, use_container_width=True)
