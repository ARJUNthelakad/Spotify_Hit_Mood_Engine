
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Spotify Hit & Mood Engine",
    page_icon="🎧",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background-color: #171717;
}
div[data-testid="stMetric"] {
    background-color: #242424;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    return (
        joblib.load("models/boosting_model.pkl"),
        joblib.load("models/logistic_model.pkl"),
        joblib.load("models/knn_model.pkl"),
        joblib.load("models/mood_scaler.pkl"),
        joblib.load("models/mood_pca.pkl"),
        joblib.load("models/kmeans_model.pkl")
    )


@st.cache_data
def load_dataset():

    data = pd.read_csv("dataset/spotify_tracks.csv")

    data = data.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )

    required_columns = [
        "track_id", "track_name", "artists",
        "album_name", "track_genre", "popularity",
        "duration_ms", "explicit", "danceability",
        "energy", "key", "loudness", "mode",
        "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence",
        "tempo", "time_signature"
    ]

    data = data.dropna(
        subset=required_columns
    ).drop_duplicates()

    data["explicit"] = data["explicit"].astype(int)

    return data


try:

    (
        boosting_model,
        logistic_model,
        knn_model,
        mood_scaler,
        mood_pca,
        kmeans_model
    ) = load_models()

    df = load_dataset()

except Exception as e:

    st.error(f"Could not load project files: {e}")
    st.stop()


# ==========================================
# 3. FEATURES AND MOOD LABELS
# ==========================================

hit_features = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature"
]

mood_features = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "tempo",
    "loudness"
]

MOOD_NAMES = {
    0: "Live / Performance-Oriented",
    1: "Intense / High-Energy",
    2: "Calm / Instrumental",
    3: "Relaxed / Acoustic",
    4: "Happy / Dance-Oriented"
}

MOOD_DESCRIPTIONS = {
    0: "Live / Performance-Oriented 🎤",
    1: "Intense / High-Energy ⚡",
    2: "Calm / Instrumental 🌙",
    3: "Relaxed / Acoustic 🌿",
    4: "Happy / Dance-Oriented 💃"
}

CLUSTER_COLORS = {
    0: "#00CC96",
    1: "#EF553B",
    2: "#AB63FA",
    3: "#FFA15A",
    4: "#636EFA"
}

feature_defaults = df[hit_features].median().to_dict()


# ==========================================
# 4. LANGUAGE DETECTION
# ==========================================

def detect_language(row):

    text = (
        str(row["track_name"]) + " " +
        str(row["album_name"]) + " " +
        str(row["track_genre"])
    ).lower()

    if "telugu" in text:
        return "Telugu"
    elif "tamil" in text:
        return "Tamil"
    elif "hindi" in text or "bollywood" in text:
        return "Hindi"
    elif "malayalam" in text:
        return "Malayalam"
    elif "punjabi" in text:
        return "Punjabi"
    elif "kannada" in text:
        return "Kannada"
    elif "spanish" in text or "latin" in text:
        return "Spanish"
    elif "korean" in text or "k-pop" in text:
        return "Korean"

    return "Other / not specified"


# ==========================================
# 5. PREPARE SONG CATALOGUE
# ==========================================

@st.cache_data
def prepare_catalog(
    data,
    _boosting_model,
    _mood_scaler,
    _mood_pca,
    _kmeans_model
):

    catalog = data.copy()

    probabilities = _boosting_model.predict_proba(
        catalog[hit_features]
    )[:, 1]

    catalog["Hit score (%)"] = np.round(
        probabilities * 100
    ).astype(int)

    scaled = _mood_scaler.transform(
        catalog[mood_features]
    )

    catalog["cluster_id"] = _kmeans_model.predict(
        scaled
    )

    catalog["Mood"] = catalog["cluster_id"].map(
        MOOD_DESCRIPTIONS
    )

    coordinates = _mood_pca.transform(scaled)

    catalog["PC1"] = coordinates[:, 0]
    catalog["PC2"] = coordinates[:, 1]

    catalog["Language"] = catalog.apply(
        detect_language,
        axis=1
    )

    catalog["Spotify"] = (
        "https://open.spotify.com/track/"
        + catalog["track_id"].astype(str)
    )

    return catalog


with st.spinner("Preparing Spotify song catalogue..."):

    catalog_df = prepare_catalog(
        df,
        boosting_model,
        mood_scaler,
        mood_pca,
        kmeans_model
    )


# ==========================================
# 6. SIDEBAR INPUTS
# ==========================================

st.sidebar.title("🎛️ Track Audio Features")

st.sidebar.caption(
    "Adjust the characteristics of your song."
)

danceability = st.sidebar.slider(
    "Danceability", 0.0, 1.0, 0.55, 0.01
)

energy = st.sidebar.slider(
    "Energy", 0.0, 1.0, 0.75, 0.01
)

valence = st.sidebar.slider(
    "Valence (Positivity)", 0.0, 1.0, 0.60, 0.01
)

acousticness = st.sidebar.slider(
    "Acousticness", 0.0, 1.0, 0.20, 0.01
)

instrumentalness = st.sidebar.slider(
    "Instrumentalness", 0.0, 1.0, 0.10, 0.01
)

loudness = st.sidebar.slider(
    "Loudness (dB)", -40.0, 0.0, -6.0, 0.1
)

tempo = st.sidebar.slider(
    "Tempo (BPM)", 40.0, 220.0, 120.0, 1.0
)

speechiness = st.sidebar.slider(
    "Speechiness", 0.0, 1.0, 0.08, 0.01
)

liveness = st.sidebar.slider(
    "Liveness", 0.0, 1.0, 0.15, 0.01
)

duration_sec = st.sidebar.slider(
    "Duration (seconds)", 60, 420, 210
)

explicit = st.sidebar.checkbox(
    "Explicit Lyrics",
    value=False
)


# ==========================================
# 7. HEADER AND TABS
# ==========================================

st.title("🎧 Spotify Hit & Mood Engine")

st.caption(
    "Hit = popularity ≥ 60 | "
    "Classifier: Tuned HistGradientBoosting"
)

tab1, tab2, tab3 = st.tabs([
    "🎵 Predict a Song",
    "📊 Model Results",
    "🔎 Discover Songs"
])


# ==========================================
# TAB 1: PREDICT A SONG
# ==========================================

with tab1:

    # Build input
    input_values = feature_defaults.copy()

    input_values.update({
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "loudness": loudness,
        "tempo": tempo,
        "speechiness": speechiness,
        "liveness": liveness,
        "duration_ms": duration_sec * 1000,
        "explicit": int(explicit)
    })

    input_data = pd.DataFrame(
        [[input_values[f] for f in hit_features]],
        columns=hit_features
    )

    mood_input = pd.DataFrame(
        [[input_values[f] for f in mood_features]],
        columns=mood_features
    )

    # Hit prediction
    hit_probability = float(
        boosting_model.predict_proba(input_data)[0][1]
    )

    hit_prediction = int(
        boosting_model.predict(input_data)[0]
    )

    # Mood prediction
    mood_scaled = mood_scaler.transform(mood_input)

    cluster_id = int(
        kmeans_model.predict(mood_scaled)[0]
    )

    # PCA coordinates for user's song
    mood_coordinates = mood_pca.transform(mood_scaled)

    user_pc1 = float(mood_coordinates[0, 0])
    user_pc2 = float(mood_coordinates[0, 1])

    # --------------------------------------
    # PREDICTION DISPLAY
    # --------------------------------------

    left, right = st.columns([0.9, 1.3], gap="large")

    with left:

        st.subheader("🎧 Audio Features")

        st.metric("Danceability", f"{danceability:.2f}")
        st.metric("Energy", f"{energy:.2f}")
        st.metric("Valence", f"{valence:.2f}")
        st.metric("Tempo", f"{tempo:.0f} BPM")
        st.metric("Duration", f"{duration_sec} seconds")

    with right:

        st.subheader("🎯 Prediction")

        c1, c2 = st.columns(2)

        c1.metric(
            "Hit probability",
            f"{hit_probability * 100:.1f}%"
        )

        c2.metric(
            "Mood cluster",
            f"Cluster {cluster_id}"
        )

        st.progress(hit_probability)

        if hit_prediction == 1:
            st.success("🔥 Likely to be a Hit!")
        else:
            st.info("Probably Not a Hit")

        st.write("**Predicted Mood**")

        st.markdown(
            f"### {MOOD_DESCRIPTIONS[cluster_id]}"
        )

        st.caption(
            "Other unspecified hit-model features use dataset medians."
        )

    # --------------------------------------
    # HIGH QUALITY PCA GRAPH
    # --------------------------------------

    st.divider()

    st.subheader("📊 Spotify Mood Cluster Visualization")

    st.caption(
        "The golden star marks the position of your input song."
    )

    # Sample up to 4000 songs from each cluster
    sample_parts = []

    for cluster in range(5):

        cluster_data = catalog_df[
            catalog_df["cluster_id"] == cluster
        ]

        if not cluster_data.empty:

            sample_parts.append(
                cluster_data.sample(
                    n=min(4000, len(cluster_data)),
                    random_state=42
                )
            )

    sample_plot = pd.concat(
        sample_parts,
        ignore_index=True
    )

    fig = go.Figure()

    # Add five mood clusters
    for cluster in range(5):

        sub = sample_plot[
            sample_plot["cluster_id"] == cluster
        ]

        fig.add_trace(
            go.Scattergl(
                x=sub["PC1"],
                y=sub["PC2"],
                mode="markers",
                name=MOOD_NAMES[cluster],

                text=(
                    sub["track_name"].astype(str)
                    + " — "
                    + sub["artists"].astype(str)
                ),

                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "PCA 1: %{x:.3f}<br>"
                    "PCA 2: %{y:.3f}"
                    "<extra></extra>"
                ),

                marker=dict(
                    color=CLUSTER_COLORS[cluster],
                    size=5,
                    opacity=0.65
                )
            )
        )

    # --------------------------------------
    # YOUR SONG: PROMINENT GOLDEN STAR
    # --------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[user_pc1],
            y=[user_pc2],

            mode="markers+text",

            name="⭐ YOUR SONG",

            text=["YOUR SONG"],

            textposition="top center",

            textfont=dict(
                color="#FFD700",
                size=17,
                family="Arial Black"
            ),

            marker=dict(
                color="#FFD700",
                size=34,
                symbol="star",
                opacity=1,
                line=dict(
                    color="#FFFFFF",
                    width=3
                )
            ),

            showlegend=True,

            hovertemplate=(
                "<b>⭐ YOUR SONG</b><br>"
                "PCA 1: %{x:.3f}<br>"
                "PCA 2: %{y:.3f}"
                "<extra></extra>"
            ),

            cliponaxis=False
        )
    )

    # --------------------------------------
    # GRAPH LAYOUT
    # --------------------------------------

    fig.update_layout(

        title=dict(
            text="2D PCA Projection of Spotify Mood Clusters",
            x=0.5,
            xanchor="center",
            font=dict(size=21)
        ),

        template="plotly_dark",

        height=780,

        autosize=True,

        xaxis=dict(
            title="Principal Component 1",
            showgrid=True,
            zeroline=True,
            showline=True
        ),

        yaxis=dict(
            title="Principal Component 2",
            showgrid=True,
            zeroline=True,
            showline=True
        ),

        legend=dict(
            title="Mood Categories",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=12)
        ),

        margin=dict(
            l=60,
            r=230,
            t=85,
            b=60
        ),

        hovermode="closest"
    )

    # Render graph
    st.plotly_chart(
        fig,
        use_container_width=True,

        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,

            "toImageButtonOptions": {
                "format": "png",
                "filename": "spotify_mood_clusters",
                "scale": 4
            }
        }
    )

    st.caption(
        "Each point represents a song. The golden star represents "
        "your input song. PCA projects nine audio features into "
        "two dimensions."
    )


# ==========================================
# TAB 2: MODEL RESULTS
# ==========================================

with tab2:

    st.header("📊 Model Performance Comparison")

    st.write(
        "Comparison of supervised learning models "
        "evaluated on the test dataset."
    )

    results_df = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "K-Nearest Neighbors",
            "HistGradientBoosting"
        ],

        "Accuracy": [
            0.5341,
            0.8563,
            0.7243
        ],

        "Hit F1 Score": [
            0.2757,
            0.3082,
            0.4022
        ]
    })

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Hit F1 Score Comparison")

    performance_fig = px.bar(
        results_df,
        x="Model",
        y="Hit F1 Score",
        color="Model",
        text="Hit F1 Score",
        title="Classification Model Performance"
    )

    performance_fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        yaxis_range=[0, 0.5],
        height=450
    )

    st.plotly_chart(
        performance_fig,
        use_container_width=True
    )

    st.success(
        "Best model: HistGradientBoosting | "
        "Test Hit F1 Score: 0.4022"
    )

    st.write("**Best Boosting Parameters**")

    st.code("""
learning_rate = 0.1
max_iter = 150
max_leaf_nodes = 63
min_samples_leaf = 20
l2_regularization = 0.1
    """)

    st.caption(
        "Best 5-fold cross-validation F1 score: 0.4042."
    )

    st.divider()

    st.subheader("Mood Clustering Summary")

    cluster_profile = (
        catalog_df.groupby("Mood")[mood_features]
        .mean()
        .round(3)
        .reset_index()
    )

    st.dataframe(
        cluster_profile,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "PCA explained variance in two dimensions: 47.78%."
    )

    st.subheader("Mood Cluster Distribution")

    cluster_counts = (
        catalog_df["Mood"]
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = [
        "Mood",
        "Number of Songs"
    ]

    distribution_fig = px.bar(
        cluster_counts,
        x="Mood",
        y="Number of Songs",
        color="Mood",
        text="Number of Songs",
        title="Songs Across Five Mood Clusters"
    )

    distribution_fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        height=450
    )

    st.plotly_chart(
        distribution_fig,
        use_container_width=True
    )


# ==========================================
# TAB 3: DISCOVER SONGS
# ==========================================

with tab3:

    st.header("🔎 Discover Spotify Songs")

    st.write(
        "Find songs using mood, genre, language "
        "and hit-score filters."
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        mood_options = [
            "All"
        ] + list(MOOD_DESCRIPTIONS.values())

        selected_mood = st.selectbox(
            "Mood Category",
            mood_options
        )

    with f2:

        genre_options = [
            "All"
        ] + sorted(
            catalog_df["track_genre"].unique().tolist()
        )

        selected_genre = st.selectbox(
            "Genre Category",
            genre_options
        )

    with f3:

        language_options = [
            "Any"
        ] + sorted(
            catalog_df["Language"].unique().tolist()
        )

        selected_language = st.selectbox(
            "Language",
            language_options
        )

    st.divider()

    s1, s2, s3 = st.columns([2, 1.2, 1])

    with s1:

        search_query = st.text_input(
            "Search song or artist",
            ""
        )

    with s2:

        sort_by = st.radio(
            "Sort by",
            [
                "Highest hit score",
                "Most popular"
            ]
        )

    with s3:

        limit_rows = st.number_input(
            "Number of songs",
            min_value=5,
            max_value=200,
            value=25,
            step=5
        )

    # Apply filters
    filtered = catalog_df.copy()

    if selected_mood != "All":

        filtered = filtered[
            filtered["Mood"] == selected_mood
        ]

    if selected_genre != "All":

        filtered = filtered[
            filtered["track_genre"] == selected_genre
        ]

    if selected_language != "Any":

        filtered = filtered[
            filtered["Language"] == selected_language
        ]

    if search_query.strip():

        query = search_query.strip()

        mask = (
            filtered["track_name"].str.contains(
                query,
                case=False,
                na=False
            )
            |
            filtered["artists"].str.contains(
                query,
                case=False,
                na=False
            )
        )

        filtered = filtered[mask]

    sort_column = (
        "Hit score (%)"
        if sort_by == "Highest hit score"
        else "popularity"
    )

    filtered = (
        filtered.sort_values(
            by=[sort_column, "popularity"],
            ascending=False
        )
        .drop_duplicates(
            subset=["track_name", "artists"]
        )
        .head(int(limit_rows))
    )

    st.metric(
        "Songs Found",
        len(filtered)
    )

    display_df = filtered[[
        "track_name",
        "artists",
        "track_genre",
        "Mood",
        "Language",
        "popularity",
        "Hit score (%)",
        "Spotify"
    ]].rename(columns={
        "track_name": "Song",
        "artists": "Artist",
        "track_genre": "Genre",
        "popularity": "Popularity"
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,

        column_config={
            "Spotify": st.column_config.LinkColumn(
                "Spotify",
                display_text="Open ▶"
            )
        }
    )


# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "Spotify Hit & Mood Engine | Team 8 | "
    "Python • Scikit-learn • Streamlit"
)
