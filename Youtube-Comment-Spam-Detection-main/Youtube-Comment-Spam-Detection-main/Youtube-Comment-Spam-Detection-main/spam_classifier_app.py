import streamlit as st
import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

st.set_page_config(page_title="YouTube Spam Classifier with ML & Big Data", layout="wide")

st.markdown('<h1 style="text-align:center;">🎯 Made by Yashwanth & Kiran </h1>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_training_data(path="labeled_comments.csv"):
    # Load your labeled training data here - you need this file ready for ML training
    df = pd.read_csv(path)
    df = df.dropna(subset=["CONTENT", "LABEL"])
    return df

@st.cache_resource(show_spinner=False)
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['CONTENT'], df['LABEL'], test_size=0.2, random_state=42
    )
    model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

def classify_comments(model, comments):
    return model.predict(comments)

def plot_class_distribution(df):
    counts = df['PREDICTED_CLASS'].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax, color=['red', 'green'])
    ax.set_title("Predicted Comment Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# === Load and train model section ===
st.sidebar.header("Model Training & Info")

with st.sidebar.expander("⚙️ Train / Load Model"):
    st.write("This app uses a ML model trained on labeled comments.")
    st.write("Ensure you have a `labeled_comments.csv` file with columns: `CONTENT`, `LABEL` (spam/not_spam).")
    
    uploaded_training = st.file_uploader("Upload labeled training CSV", type=["csv"])
    
    if uploaded_training:
        df_train = pd.read_csv(uploaded_training)
        if 'CONTENT' in df_train.columns and 'LABEL' in df_train.columns:
            st.success("Training data loaded!")
            model, acc = train_model(df_train)
            st.write(f"Model trained! Accuracy on test set: **{acc:.2%}**")
            st.session_state['model'] = model
        else:
            st.error("Training file must contain 'CONTENT' and 'LABEL' columns.")
    else:
        st.info("Upload labeled data to train the model.")

# === Classify uploaded comments ===
st.header("📤 Upload YouTube Comments CSV to Classify")

uploaded_file = st.file_uploader("Upload CSV file with `CONTENT` column", type=["csv"], key="comments_upload")

if uploaded_file:
    st.info("Loading comments with Dask for efficient processing...")
    # Load large CSV with dask
    df_dask = dd.read_csv(uploaded_file)
    
    if "CONTENT" not in df_dask.columns:
        st.error("❌ Uploaded file must have a 'CONTENT' column.")
    else:
        # Convert to pandas (small preview) and display
        df_sample = df_dask.head(1000)  # preview first 1000 rows
        st.write("Preview of uploaded comments:")
        st.dataframe(df_sample[["CONTENT"]].head(20))
        
        # Make sure model is loaded
        if 'model' not in st.session_state:
            st.warning("Please upload training data and train the model first in the sidebar.")
        else:
            model = st.session_state['model']
            
            st.info("Classifying comments (this may take a moment for large files)...")
            
            # Classify all comments with Dask map_partitions and pandas apply
            def classify_partition(df_partition):
                df_partition["PREDICTED_CLASS"] = model.predict(df_partition["CONTENT"])
                return df_partition
            
            df_classified = df_dask.map_partitions(classify_partition).compute()
            
            st.success(f"Classification done! {len(df_classified)} comments processed.")
            
            # Show distribution plot
            plot_class_distribution(df_classified)
            
            # Show table preview
            st.dataframe(df_classified[["CONTENT", "PREDICTED_CLASS"]].head(20))
            
            # Provide downloads
            csv_all = df_classified.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download All Classified Comments", csv_all, "classified_comments.csv", "text/csv")
            
            spam_df = df_classified[df_classified["PREDICTED_CLASS"] == "spam"]
            spam_csv = spam_df.to_csv(index=False).encode('utf-8')
            st.download_button("🔥 Download SPAM Comments Only", spam_csv, "spam_comments.csv", "text/csv")
            
            not_spam_df = df_classified[df_classified["PREDICTED_CLASS"] == "not_spam"]
            not_spam_csv = not_spam_df.to_csv(index=False).encode('utf-8')
            st.download_button("✅ Download NOT SPAM Comments Only", not_spam_csv, "not_spam_comments.csv", "text/csv")

# === Manual comment test ===
st.markdown("---")
st.header("🧪 Test a Comment Manually")

user_input = st.text_input("Enter a comment to classify:")

if user_input:
    if 'model' in st.session_state:
        model = st.session_state['model']
        pred = model.predict([user_input])[0]
        if pred == "spam":
            st.markdown("<h2 style='color:red;'>❌ SPAM COMMENT</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>✅ NOT SPAM COMMENT</h2>", unsafe_allow_html=True)
    else:
        st.warning("Please upload training data and train the model first.")

st.markdown("---")
st.markdown("Made by Yashwanth & Kiran | Upgraded with ML & Big Data support")
