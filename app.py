# =========================================================
# 📌 streamlit_app.py — Customer Support AI Dashboard
# =========================================================

import re
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------------------
# 1️⃣ PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Customer Support AI Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------------------
# 2️⃣ SIDEBAR NAVIGATION
# ---------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 EDA Dashboard", "🤖 RAG Query Assistant", "🚨 Urgency Classifier"]
)

# ---------------------------
# 3️⃣ LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    """Load customer support tickets data with error handling"""
    try:
        csv_path = "customer_support_tickets_cleaned.csv"
        if not os.path.exists(csv_path):
            st.error(f"❌ Data file not found: {csv_path}")
            st.stop()
        df = pd.read_csv(csv_path)
        if df.empty:
            st.warning("⚠️ The data file is empty.")
            st.stop()
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to initialize app: {str(e)}")
    st.stop()

# ---------------------------
# 4️⃣ BASIC CLEAN FUNCTION
# ---------------------------
def basic_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ---------------------------
# 5️⃣ URGENCY CLASSIFICATION HELPER
# ---------------------------
def classify_and_display(priority):
    """Classify priority and display result"""
    priority_lower = str(priority).lower().strip()
    urgent_priorities = ['critical', 'high']
    non_urgent_priorities = ['medium', 'low']
    
    if priority_lower in urgent_priorities:
        st.success("🔥 **URGENT TICKET**")
        st.error(f"⚠️ Priority: **{priority.upper()}**")
        st.warning("💡 **Action Required:** This ticket requires immediate attention!")
        st.write("""
        **Recommended Actions:**
        - Escalate to senior support team
        - Respond within 1-2 hours
        - Monitor closely until resolved
        - Update customer frequently
        """)
    elif priority_lower in non_urgent_priorities:
        st.info("✅ **NON-URGENT TICKET**")
        st.success(f"📋 Priority: **{priority.capitalize()}**")
        st.write("💡 **Action Required:** This ticket can be handled with standard priority.")
        st.write("""
        **Recommended Actions:**
        - Process in regular queue
        - Respond within 24-48 hours
        - Follow standard procedures
        - Update customer as needed
        """)
    else:
        st.warning(f"⚠️ **UNKNOWN PRIORITY: {priority}**")
        st.info("""
        This priority level is not recognized. 
        
        **Expected priorities:**
        - **Urgent:** Critical, High
        - **Not Urgent:** Medium, Low
        Please verify the data or update the classification rules.
        """)

# ---------------------------
# 6️⃣ ML MODEL FOR NEW TICKET PREDICTION (Using Pre-trained Model)
# ---------------------------
@st.cache_resource
def load_pretrained_urgency_model():
    try:
        model_path = "urgency_model_engineered.joblib"
        vectorizer_path = "urgency_vectorizer_engineered.joblib"
        features_path = "urgency_feature_cols.joblib"
        
        missing_files = [p for p in [model_path, vectorizer_path, features_path] if not os.path.exists(p)]
        if missing_files:
            return None, None, None, f"Missing model files: {', '.join(missing_files)}"
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        feature_cols = joblib.load(features_path)
        return model, vectorizer, feature_cols, "Pre-trained model loaded successfully!"
    except Exception as e:
        return None, None, None, f"Error loading model: {str(e)}"

def create_features(texts):
    df_feat = pd.DataFrame({"text": texts})
    df_feat["char_count"] = df_feat["text"].apply(len)
    df_feat["word_count"] = df_feat["text"].apply(lambda x: len(x.split()))
    df_feat["exclamation_count"] = df_feat["text"].apply(lambda x: x.count("!"))
    df_feat["question_count"] = df_feat["text"].apply(lambda x: x.count("?"))
    df_feat["desc_length"] = df_feat["text"].apply(len)
    df_feat["uppercase_word_count"] = df_feat["text"].apply(
        lambda x: sum(1 for word in x.split() if word.isupper() and len(word) > 1)
    )
    urgent_keywords = [
        'urgent', 'critical', 'emergency', 'asap', 'immediately', 'now',
        'help', 'down', 'broken', 'failed', 'crash', 'error', 'issue',
        'problem', 'cannot', 'unable', 'not working', 'blocked', 'stuck'
    ]
    df_feat["urgent_keyword_count"] = df_feat["text"].apply(
        lambda x: sum(keyword in x.lower() for keyword in urgent_keywords)
    )
    df_feat["has_urgent_keyword"] = (df_feat["urgent_keyword_count"] > 0).astype(int)
    return df_feat

def predict_urgency_from_description(description, model, vectorizer, feature_cols):
    cleaned = basic_clean(description)
    if not cleaned or len(cleaned) < 5:
        return None, None, "Description too short or empty"
    X_text = vectorizer.transform([cleaned])
    X_extra = create_features([description])[feature_cols]
    X_final = hstack([X_text, X_extra.values])
    prediction = model.predict(X_final)[0]
    confidence = model.predict_proba(X_final)[0][prediction]
    return prediction, confidence, None

# =========================================================
# 🧭 PAGE 1 — EDA DASHBOARD (with Data Labels)
# =========================================================
if page == "📊 EDA Dashboard":
    st.title("📊 Customer Support Ticket Dashboard")
    st.markdown("An interactive overview of ticket trends, products, channels, and resolution times.")

    try:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tickets", len(df))
        
        if 'Time to Resolution' in df.columns:
            resolved_pct = df['Time to Resolution'].notna().mean() * 100
            col2.metric("Resolved %", f"{resolved_pct:.1f}%")
        else:
            col2.metric("Resolved %", "N/A")
        
        if 'Customer Age' in df.columns and df['Customer Age'].notna().any():
            avg_age = int(df['Customer Age'].mean())
            col3.metric("Avg Customer Age", avg_age)
        else:
            col3.metric("Avg Customer Age", "N/A")

        # ✅ Top Products by Ticket Volume — with data labels
        st.subheader("🎯 Top Products by Ticket Volume")
        if 'Product Purchased' in df.columns:
            product_counts = df['Product Purchased'].value_counts().head(10)
            if not product_counts.empty:
                fig, ax = plt.subplots(figsize=(10, 4))
                product_counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_ylabel("Ticket Count")
                ax.set_xlabel("Product")
                plt.xticks(rotation=45, ha='right')

                for i, v in enumerate(product_counts.values):
                    ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No product data available.")
        else:
            st.warning("'Product Purchased' column not found in data.")

        # ✅ Ticket Status Distribution — with data labels
        st.subheader("📌 Ticket Status Distribution")
        if 'Ticket Status' in df.columns:
            status_counts = df['Ticket Status'].value_counts()
            if not status_counts.empty:
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                status_counts.plot(kind='bar', ax=ax2, color='orange')
                ax2.set_ylabel("Count")
                ax2.set_xlabel("Status")
                plt.xticks(rotation=45, ha='right')

                for i, v in enumerate(status_counts.values):
                    ax2.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("No status data available.")
        else:
            st.warning("'Ticket Status' column not found in data.")
    
    except Exception as e:
        st.error(f"❌ Error displaying dashboard: {str(e)}")

# =========================================================
# 🧭 PAGE 2 — RAG QUERY ASSISTANT
# =========================================================
elif page == "🤖 RAG Query Assistant":
    st.title("🔍 RAG Query Assistant")
    st.markdown("Ask questions like *'What are common refund issues?'* to search historical tickets.")
    # (no changes below)
    try:
        @st.cache_resource
        def build_rag():
            if 'Ticket Description' not in df.columns:
                st.error("❌ 'Ticket Description' column not found in data.")
                st.stop()
            docs = df['Ticket Description'].fillna("").tolist()
            clean_docs = [basic_clean(d) for d in docs]
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
            doc_vecs = vectorizer.fit_transform(clean_docs)
            nn = NearestNeighbors(n_neighbors=min(5, len(docs)), metric='cosine').fit(doc_vecs)
            return {"docs": docs, "vecs": doc_vecs, "vectorizer": vectorizer, "nn": nn}

        rag = build_rag()

        user_query = st.text_input("💬 Enter your query:")
        top_k = st.slider("Number of tickets to retrieve", 3, 10, 5)

        if st.button("Search"):
            if not user_query.strip():
                st.warning("⚠️ Please enter a query.")
            else:
                q_vec = rag["vectorizer"].transform([basic_clean(user_query)])
                actual_k = min(top_k, len(rag["docs"]))
                dists, idxs = rag["nn"].kneighbors(q_vec, n_neighbors=actual_k)
                sims = 1 - dists.flatten()
                st.subheader("📌 Retrieved Tickets")
                for i, sim in zip(idxs.flatten(), sims):
                    with st.expander(f"Ticket {i} | Similarity: {sim:.2f}"):
                        st.write(rag["docs"][i])
    except Exception as e:
        st.error(f"❌ Error initializing RAG system: {str(e)}")

# =========================================================
# 🧭 PAGE 3 — URGENCY CLASSIFIER
# =========================================================

        # --- Classification Section ---
        st.subheader("🔍 Classify a Ticket")
        
        # Three input methods
        input_method = st.radio("Choose input method:", 
                                ["Enter Ticket ID", "Select Priority Manually", "Predict from New Description"])
        
        if input_method == "Enter Ticket ID":
            if 'Ticket ID' in df.columns:
                ticket_id = st.text_input("🎫 Enter Ticket ID:")
                
                if st.button("🔍 Classify by Ticket ID"):
                    if not ticket_id.strip():
                        st.warning("⚠️ Please enter a Ticket ID.")
                    else:
                        # Find ticket in dataset
                        ticket_data = df[df['Ticket ID'].astype(str) == ticket_id.strip()]
                        
                        if ticket_data.empty:
                            st.error(f"❌ Ticket ID '{ticket_id}' not found in dataset.")
                        else:
                            ticket_info = ticket_data.iloc[0]
                            priority = ticket_info['Ticket Priority']
                            
                            # Display ticket info
                            with st.expander("📋 Ticket Details", expanded=True):
                                st.write(f"**Ticket ID:** {ticket_info['Ticket ID']}")
                                st.write(f"**Priority:** {priority}")
                                if 'Ticket Subject' in ticket_info:
                                    st.write(f"**Subject:** {ticket_info['Ticket Subject']}")
                                if 'Ticket Description' in ticket_info:
                                    st.write(f"**Description:** {ticket_info['Ticket Description']}")
                                if 'Ticket Status' in ticket_info:
                                    st.write(f"**Status:** {ticket_info['Ticket Status']}")
                            
                            # Classify based on priority
                            classify_and_display(priority)
            else:
                st.warning("⚠️ 'Ticket ID' column not found in dataset. Please use manual priority selection.")
        
        elif input_method == "Select Priority Manually":
            st.info("💡 Select a priority level to see its urgency classification.")
            
            # Get unique priorities from dataset
            available_priorities = df['Ticket Priority'].unique().tolist()
            
            if not available_priorities:
                st.error("❌ No priority values found in dataset.")
            else:
                selected_priority = st.selectbox("📌 Select Ticket Priority:", 
                                                 sorted(available_priorities))
                
                if st.button("🔮 Classify Priority"):
                    classify_and_display(selected_priority)
                    
                    # Show sample tickets with this priority
                    st.subheader(f"📑 Sample Tickets with '{selected_priority}' Priority")
                    sample_tickets = df[df['Ticket Priority'] == selected_priority].head(5)
                    
                    if 'Ticket Description' in sample_tickets.columns:
                        for idx, row in sample_tickets.iterrows():
                            with st.expander(f"Ticket: {row.get('Ticket Subject', 'N/A')}"):
                                st.write(f"**Description:** {row['Ticket Description']}")
                                if 'Ticket Status' in row:
                                    st.write(f"**Status:** {row['Ticket Status']}")
        
        else:  # Predict from New Description
            st.info("🤖 Enter a ticket description to predict urgency using pre-trained machine learning model.")
            
            # Load pre-trained model
            with st.spinner("🔄 Loading pre-trained ML model..."):
                model, vectorizer, feature_cols, message = load_pretrained_urgency_model()
            
            if model is None:
                st.error(f"❌ Unable to load model: {message}")
                st.info("Please ensure the following files are present:")
                st.code("""
- urgency_model_engineered.joblib
- urgency_vectorizer_engineered.joblib
- urgency_feature_cols.joblib
                """)
            else:
                st.success(f"✅ {message}")
                
                # Input for new ticket
                st.markdown("---")
                new_description = st.text_area(
                    "📝 Enter New Ticket Description:",
                    height=150,
                    placeholder="Example: My account has been locked and I cannot access critical financial data. This is blocking our entire team's work. Need urgent help!"
                )
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    predict_button = st.button("🔮 Predict Urgency", type="primary")
                
                if predict_button:
                    if not new_description.strip():
                        st.warning("⚠️ Please enter a ticket description.")
                    else:
                        with st.spinner("🤔 Analyzing ticket..."):
                            prediction, confidence, error = predict_urgency_from_description(
                                new_description, model, vectorizer, feature_cols
                            )
                        
                        if error:
                            st.error(f"❌ {error}")
                        else:
                            st.markdown("---")
                            st.subheader("📊 Prediction Results")
                            
                            # Display prediction
                            if prediction == 1:
                                st.success("🔥 **URGENT TICKET PREDICTED**")
                                st.error(f"⚠️ **Confidence: {confidence:.1%}**")
                                st.warning("💡 **Recommendation:** This ticket likely requires immediate attention!")
                                
                                st.write("""
                                **Suggested Actions:**
                                - Treat as high priority
                                - Escalate to senior support team
                                - Respond within 1-2 hours
                                - Monitor closely until resolved
                                - Keep customer updated frequently
                                """)
                                
                                # Show similar urgent tickets from dataset
                                st.markdown("---")
                                st.subheader("📋 Similar Urgent Tickets from History")
                                urgent_tickets = df[df['Ticket Priority'].isin(['Critical', 'High'])]
                                if not urgent_tickets.empty:
                                    sample = urgent_tickets.sample(min(3, len(urgent_tickets)))
                                    for idx, row in sample.iterrows():
                                        with st.expander(f"Example: {row.get('Ticket Subject', 'N/A')[:50]}..."):
                                            st.write(f"**Priority:** {row['Ticket Priority']}")
                                            st.write(f"**Description:** {row['Ticket Description'][:200]}...")
                            
                            else:
                                st.info("✅ **NON-URGENT TICKET PREDICTED**")
                                st.success(f"📊 **Confidence: {confidence:.1%}**")
                                st.write("💡 **Recommendation:** This ticket can be handled with standard priority.")
                                
                                st.write("""
                                **Suggested Actions:**
                                - Add to regular support queue
                                - Respond within 24-48 hours
                                - Follow standard procedures
                                - Update customer as needed
                                """)
    
    except Exception as e:
        st.error(f"❌ Error in Urgency Classifier: {str(e)}")
dont change anything i just want the charts in the dashboard to have data labels
