import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel


st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/processed_netflix.csv")

@st.cache_resource
def load_models():
    tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
    kmeans = pickle.load(open("models/kmeans.pkl", "rb"))
    return tfidf, kmeans

df = load_data()
tfidf, kmeans = load_models()



X = tfidf.transform(df["combined_features"])
similarity_matrix = linear_kernel(X, X)

indices = pd.Series(
    df.index,
    index=df["title"].astype(str).str.strip()
).drop_duplicates()
def recommend(title, n=5):

    title = title.strip()

    if title not in indices.index:
        return None

    idx = indices[title]

    sim_scores = list(
        enumerate(similarity_matrix[idx])
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:n+1]

    show_indices = [i[0] for i in sim_scores]

    return df.iloc[show_indices][
        [
            "title",
            "listed_in",
            "release_year",
            "rating"
        ]
    ]


st.sidebar.title("Netflix Analytics")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Recommendations",
        "Cluster Analytics"
    ]
)


if page == "Home":

    st.title("🎬 Netflix Show Clustering & Recommendation System")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Titles",
            len(df)
        )

    with col2:
        st.metric(
            "Total Clusters",
            df["cluster"].nunique()
        )

    with col3:
        st.metric(
            "Movies",
            len(df[df["type"] == "Movie"])
        )

    st.markdown("---")

    st.subheader("Dataset Overview")

    st.dataframe(
        df.head(10)
    )


elif page == "Recommendations":

    st.title("🎥 Netflix Recommendations")

    selected_title = st.selectbox(
        "Choose a Netflix Title",
        sorted(df["title"].dropna().unique())
    )

    selected_data = df[
        df["title"] == selected_title
    ].iloc[0]

    st.subheader("Selected Show")

    st.write(
        f"**Genre:** {selected_data['listed_in']}"
    )

    st.write(
        f"**Release Year:** {selected_data['release_year']}"
    )

    st.write(
        f"**Rating:** {selected_data['rating']}"
    )

    st.write(
        f"**Cluster:** {selected_data['cluster']}"
    )

    st.write(
        f"**Description:** {selected_data['description']}"
    )

    if st.button("Get Recommendations"):

        recommendations = recommend(
            selected_title,
            n=5
        )

        st.subheader(
            "Recommended Shows"
        )

        st.dataframe(
            recommendations,
            use_container_width=True
        )


elif page == "Cluster Analytics":

    st.title("📊 Cluster Analytics")

    st.subheader(
        "Cluster Distribution"
    )

    cluster_counts = (
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        cluster_counts
    )

    st.subheader(
        "Titles Per Cluster"
    )

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(df["cluster"].unique())
    )

    cluster_titles = df[
        df["cluster"] == selected_cluster
    ][
        [
            "title",
            "listed_in",
            "release_year"
        ]
    ]

    st.dataframe(
        cluster_titles.head(50),
        use_container_width=True
    )

    st.subheader(
        "Top Genres in Cluster"
    )

    genres = (
        df[df["cluster"] == selected_cluster]
        ["listed_in"]
        .value_counts()
        .head(10)
    )

    st.write(genres)


