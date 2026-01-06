import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Spam Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Load Model & Data ----------------
try:
    pipeline = pickle.load(open("models/spam_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model file 'spam_model.pkl' not found.")
    st.stop()

try:
    df = pd.read_csv("data/raw/spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
except FileNotFoundError:
    st.error("Dataset 'spam.csv' not found.")
    st.stop()

# ---------------- Metrics (Test Set) ----------------
X = df["message"]
y = df["label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ---------------- Header ----------------
st.title("Spam vs Ham Detection System")
st.write("TF-IDF + Multinomial Naive Bayes | Production-Ready NLP Classifier")
st.divider()

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Data Overview", "Model Performance", "Make Predictions", "Model Insights"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
st.sidebar.write("Algorithm: Multinomial Naive Bayes")
st.sidebar.write("Vectorizer: TF-IDF")
st.sidebar.write("Features: 3000")

# ================= HOME =================
if page == "Home":
    st.subheader("About This Application")
    st.write(
        "This application detects whether a message is **Spam** or **Ham** using "
        "Natural Language Processing and a probabilistic classifier."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Messages", f"{len(df):,}")
    col2.metric("Spam Messages", int(df["label"].sum()))
    col3.metric("Ham Messages", int((df["label"] == 0).sum()))
    col4.metric("Test Accuracy", f"{acc:.2%}")

    st.markdown("---")
    st.subheader("Sample Messages")
    st.dataframe(df.sample(10, random_state=42))

# ================= DATA OVERVIEW =================
elif page == "Data Overview":
    st.header("Dataset Analysis")

    col1, col2 = st.columns(2)

    with col1:
        label_counts = df["label"].value_counts().rename({0: "Ham", 1: "Spam"})
        fig = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            title="Class Distribution",
            hole=0.3
        )
        st.plotly_chart(fig)

    with col2:
        df["msg_length"] = df["message"].str.len()
        fig = px.histogram(
            df,
            x="msg_length",
            color=df["label"].map({0: "Ham", 1: "Spam"}),
            nbins=50,
            title="Message Length Distribution"
        )
        st.plotly_chart(fig)

# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":
    st.header("Model Performance (Test Set)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")

    st.markdown("---")

    st.subheader("Confusion Matrix")
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Ham", "Spam"],
        y=["Ham", "Spam"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues"
    ))
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    st.plotly_chart(fig)

    st.subheader("Classification Report")
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"],
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df, use_container_width=True)

# ================= PREDICTION =================
elif page == "Make Predictions":
    st.header("Predict Message Type")

    text = st.text_area("Enter a message:")

    if st.button("Predict", type="primary"):
        if text.strip() == "":
            st.warning("Please enter a message.")
        else:
            pred = pipeline.predict([text])[0]
            prob = pipeline.predict_proba([text])[0]

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                if pred == 1:
                    st.error("SPAM MESSAGE")
                else:
                    st.success("HAM MESSAGE")

            with col2:
                st.metric("Spam Probability", f"{prob[1]:.2%}")
                st.metric("Ham Probability", f"{prob[0]:.2%}")

            fig = go.Figure(go.Bar(
                x=["Ham", "Spam"],
                y=[prob[0], prob[1]],
                text=[f"{p:.2%}" for p in prob],
                textposition="outside"
            ))
            fig.update_layout(
                title="Prediction Confidence",
                yaxis_title="Probability",
                height=400
            )
            st.plotly_chart(fig)

# ================= MODEL INSIGHTS =================
elif page == "Model Insights":
    st.header("Model Explainability")

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["model"]

    feature_names = np.array(tfidf.get_feature_names_out())
    spam_words = feature_names[np.argsort(clf.feature_log_prob_[1])[-20:]]
    ham_words = feature_names[np.argsort(clf.feature_log_prob_[0])[-20:]]

    spam_df = pd.DataFrame({
        "Rank": range(1, len(spam_words) + 1),
        "Word": list(reversed(spam_words))
    })

    ham_df = pd.DataFrame({
        "Rank": range(1, len(ham_words) + 1),
        "Word": list(reversed(ham_words))
    })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Spam Indicators")
        st.dataframe(spam_df, use_container_width=True)

    with col2:
        st.subheader("Top Ham Indicators")
        st.dataframe(ham_df, use_container_width=True)
    st.write('After addressing class imbalance via threshold tuning, the model achieves a spam recall of over 97%, ensuring minimal spam leakage. Although precision decreases, this trade-off is acceptable for spam detection where false negatives are costlier than false positives')
# ---------------- Footer ----------------
def about_the_coder():
    # We use a non-indented string to prevent Markdown from treating it as code
    html_code = """
    <style>
    .coder-card {
        background-color: transparent;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 10px;
        padding: 20px;
        display: flex;
        align-items: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .coder-img {
        width: 100px; /* Slightly larger for better visibility */
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #FF4B4B; /* Streamlit Red */
        margin-right: 25px;
        flex-shrink: 0; /* Prevents image from shrinking */
    }
    .coder-info h3 {
        margin: 0;
        font-family: 'Source Sans Pro', sans-serif;
        color: inherit;
        font-size: 1.4rem;
        font-weight: 600;
    }
    .coder-info p {
        margin: 10px 0;
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.5;
    }
    .social-links {
        margin-top: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
    }
    .social-links a {
        text-decoration: none;
        color: #FF4B4B;
        font-weight: bold;
        font-size: 0.95rem;
        transition: color 0.3s;
    }
    .social-links a:hover {
        color: #ff2b2b;
        text-decoration: underline;
    }
    /* Mobile responsiveness */
    @media (max-width: 600px) {
        .coder-card {
            flex-direction: column;
            text-align: center;
            padding: 15px;
        }
        .coder-img {
            margin-right: 0;
            margin-bottom: 15px;
            width: 80px;
            height: 80px;
        }
        .social-links {
            justify-content: center;
        }
    }
    </style>  
    <div class="coder-card">
        <img src="https://ui-avatars.com/api/?name=Yash+Vasudeva&size=120&background=FF4B4B&color=fff&bold=true&rounded=true" class="coder-img" alt="Yash Vasudeva"/>
        <div class="coder-info">
            <h3>Developed by Yash Vasudeva</h3>
            <p>
                Results-driven Data & AI Professional skilled in <b>Data Analytics</b>, 
                <b>Machine Learning</b>, and <b>Deep Learning</b>. 
                Passionate about transforming raw data into business value and building intelligent solutions.
            </p>
            <div class="social-links">
                <a href="https://www.linkedin.com/in/yash-vasudeva/" target="_blank">LinkedIn</a>
                <a href="https://github.com/yashvasudeva1" target="_blank">GitHub</a>
                <a href="mailto:vasudevyash@gmail.com">Contact</a>
                <a href="https://yashvasudeva.vercel.app/" target="_blank">Portfolio</a>
            </div>
        </div>
    </div>
    """
        
    st.markdown(html_code, unsafe_allow_html=True)

st.divider()

if __name__ == "__main__":
    about_the_coder()