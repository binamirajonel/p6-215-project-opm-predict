#!/usr/bin/env python
# coding: utf-8

# # GTZAN Music Genre Classification Dataset
# 
# 
# ## Contents
# 
# - **Genres Original**  
#   A collection of 1,000 audio tracks across 10 music genres (100 tracks per genre), each lasting 30 seconds. This is the well-known **GTZAN dataset**, often dubbed the "MNIST of sound."
# 
# - **Images Original**  
#   Each audio file has been converted into a visual representation using **Mel Spectrograms**. Since neural networks—especially **Convolutional Neural Networks (CNNs)**—are typically designed for image input, this transformation enables effective classification using these models.
# 
# - **Two CSV Files**  
#   1. **30-Second Features File**: Contains the mean and variance of multiple audio features extracted from each full 30-second song.  
#   2. **3-Second Segments File**: Similar structure, but with songs split into 3-second segments, increasing the dataset size tenfold. More data typically leads to better performance in machine learning models.
# 
# ## Acknowledgements
# 
# The **GTZAN dataset** is one of the most frequently used public datasets in music genre recognition (MGR) research. Compiled between 2000 and 2001, it includes recordings from various sources—personal CDs, radio broadcasts, and microphones—to reflect a wide range of recording conditions.  
# More information can be found at: [http://marsyas.info/downloads/datasets.html](http://marsyas.info/downloads/datasets.html)
# 
# This was a collaborative university project. A special thanks to **James Wiltshire**, **Lauren O'Hare**, and **Minyu Lei** for being incredible teammates. The three days we spent working on this were both fun and educational.
# 

# ## Physics 215 Project: Music Genre Classification Using Machine Learning
# 
# 
# In this project, we aim to classify music genres using machine learning techniques. We will utilize the `features_30_sec.csv` file from the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data), which contains audio features extracted from 30-second clips across various music genres.
# 

# ## Load Dataset

# In[22]:

import streamlit as st
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree, plot_importance
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


st.title("OPM (Original Pilipino Music) Genre Classifier")

st.markdown("""
In this project, we aim to classify music genres using machine learning techniques. We will utilize the `features_30_sec.csv` file from the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data), which contains audio features extracted from 30-second clips across various music genres.
""")

st.header("Load Dataset")

uploaded_file = st.file_uploader("Upload feature CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Info")
    st.write(df.info())
    st.write(df.describe(include='all').T)
else:
    st.warning("Please upload the dataset to proceed.")
    st.stop()

# Preprocessing
X = df.drop(['filename','label'], axis=1)
y = df['label']

mean_cols = [col for col in X.columns if 'mfcc' in col and 'mean' in col]
var_cols = [col for col in X.columns if 'mfcc' in col and 'var' in col]
other_cols = [col for col in X.columns if col not in mean_cols + var_cols]

mean_data = X.loc[:,mean_cols]
var_data = X.loc[:,var_cols]
others_data = X.loc[:,other_cols]

# Scaling
scaler = StandardScaler()
mean_scaled = scaler.fit_transform(mean_data)
var_scaled = scaler.fit_transform(var_data)
others_scaled = scaler.fit_transform(others_data)

X_scaled = np.concatenate([others_scaled, mean_scaled, var_scaled], axis=1)

# PCA Reduction
pca1 = PCA(n_components=2).fit_transform(mean_data)
pca2 = PCA(n_components=2).fit_transform(var_data)
X_pca = np.concatenate([others_data, pca1, pca2], axis=1)

# Label Encoding
le = LabelEncoder()
y_enc = le.fit_transform(y)

use_pca = st.checkbox("Use PCA-reduced features", value=True)
X_used = X_pca if use_pca else X_scaled

X_train, X_test, y_train, y_test = train_test_split(
    X_used, y_enc, test_size=0.2, stratify=y_enc, random_state=1
)

st.header("Model Training")

st.markdown("Select a Classifier to Train")
model_choice = st.selectbox("Choose model", [
    'RandomForest', 'SVC', 'DecisionTree', 'XGBoost', 'SGD', 'NaiveBayes', 'KNN'
])

def get_model(name):
    if name == 'RandomForest':
        return RandomForestClassifier()
    elif name == 'SVC':
        return SVC()
    elif name == 'DecisionTree':
        return DecisionTreeClassifier()
    elif name == 'XGBoost':
        return XGBClassifier(eval_metric='mlogloss')
    elif name == 'SGD':
        return SGDClassifier()
    elif name == 'NaiveBayes':
        return GaussianNB()
    elif name == 'KNN':
        return KNeighborsClassifier()

if st.button("Train Model"):
    model = get_model(model_choice)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text(f"{model_choice} Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues", ax=ax)
    st.pyplot(fig)
    
st.markdown("""
            Among the evaluated classifier models, the XGBClassifier achieved the highest accuracy. Therefore, it is selected as the optimal model for this task.
            """)

###########################################    

st.header("Genre Prediction of OPM (Original Pilipino Music)")
st.markdown("""
            In this section, we will try to predict the genre of various popular OPM given its YouTube URL. It requires `yt-dlp`.
            """)
st.subheader("XGB + RFC  Ensemble Classifer on OPM")
st.markdown("""
            We used the ChatGPT language model to generate random Original Pilipino Music (OPM) across various genres. Each song was then reviewed and manually classified according to the genre categories defined in the GTZAN dataset. No classical or jazz songs were identified. Additionally, it was observed that very few songs could be distinctly categorized as blues (traditional), country, disco, metal, reggae, or rock. 
            
            Rock is particularly challenging to classify because it encompasses subgenres such as classic rock, hard rock, and alternative rock. These subgenres are often ambiguous and may be misclassified as other genres, such as pop.
            
            While many OPM tracks sonically draw inspiration from these genres, their overall production tends to lean toward pop. As a result, only five songs were selected for each of these genres. In contrast, hiphop and pop were more prevalent among the generated songs, so we included a larger number—ten for hip hop and fifteen for pop.
            """)


code = '''
import os
import subprocess
import librosa
import pandas as pd
import numpy as np
import re

def get_audio_features(y, sr):
    features = {
        'length': len(y),
        'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'chroma_stft_var': np.var(librosa.feature.chroma_stft(y=y, sr=sr)), 
        'rms_mean': np.mean(librosa.feature.rms(y=y)), 
        'rms_var': np.var(librosa.feature.rms(y=y)),
        'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)), 
        'spectral_centroid_var': np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth_mean': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)), 
        'spectral_bandwidth_var': np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)), 
        'rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'rolloff_var': np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'zero_crossing_rate_mean': np.mean(librosa.feature.zero_crossing_rate(y=y)),
        'zero_crossing_rate_var': np.var(librosa.feature.zero_crossing_rate(y=y)),
        'harmony_mean': np.mean(librosa.effects.harmonic(y)),
        'harmony_var': np.var(librosa.effects.harmonic(y)),
        'perceptr_mean': np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
        'perceptr_var': np.var(librosa.feature.spectral_contrast(y=y, sr=sr)),
        'tempo': librosa.beat.beat_track(y=y, sr=sr)[0][0],
    }

    # loop for mfcc feature:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
        features[f'mfcc{i+1}_var'] = np.var(mfcc[i])

    return features

def get_video_title(youtube_url):
    result = subprocess.run(
        ['yt-dlp', '--get-title', youtube_url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    title = result.stdout.strip()
    # Remove characters not allowed in filenames
    title = re.sub(r'[\\/*?:"<>|]', '', title)
    return title

def download_youtube_audio_and_load(
    youtube_url,
    output_dir='downloads',
    start_time='00:00:00',
    duration=30,
    sr=22050,
    trim_silence=True
):
    os.makedirs(output_dir, exist_ok=True)

    title = get_video_title(youtube_url)
    custom_filename = f"{title}.wav"
    output_filepath = os.path.join(output_dir, custom_filename)

    # Download audio starting at custom time
    download_cmd = [
        'yt-dlp',
        '-x', '--audio-format', 'wav',
        '--postprocessor-args', f'-ss {start_time} -t {duration}',
        '-o', output_filepath,
        youtube_url
    ]

    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    subprocess.run(download_cmd, check=True)

    #print(f"Audio saved as: {output_filepath}")

    # Load and optionally trim
    y, sr = librosa.load(output_filepath, sr=sr)
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=20)
        #print("Leading/trailing silence trimmed.")

    return y, sr, title

def predict_genre_for_multiple_links(urls, pca1, pca2, voting_clf, le, other_cols, mean_cols, var_cols, start_times=[]):
    results = []

    for i in range(len(urls)):
        try:
            audio_data, sample_rate, title = download_youtube_audio_and_load(
                youtube_url=urls[i],
                start_time=start_times[i] if start_times else '00:00:00'
            )

            # Feature extraction
            audio_data_feature = get_audio_features(audio_data, sample_rate)
            audio_data_df = pd.DataFrame([audio_data_feature])

            audio_data_other = audio_data_df.loc[:, other_cols]
            audio_data_mean = audio_data_df.loc[:, mean_cols]
            audio_data_var = audio_data_df.loc[:, var_cols]

            tmp1 = pca1.transform(audio_data_mean)
            tmp2 = pca2.transform(audio_data_var)

            audio_data_pca = np.concatenate([audio_data_other, tmp1, tmp2], axis=1)

            # Predict probabilities
            probs = voting_clf.predict_proba(audio_data_pca)[0]
            top2_indices = np.argsort(probs)[-2:][::-1]

            top2_genres = le.classes_[top2_indices]
            top2_probs = probs[top2_indices]

            #print(f"{title} - top genres: {top2_genres[0]} ({top2_probs[0]:.2f}), {top2_genres[1]} ({top2_probs[1]:.2f})")

            results.append({
                'title': title,
                'features': audio_data_feature,
                'predicted_genres': top2_genres.tolist(),
                'predicted_probs': top2_probs.tolist(),
                'all_probs': probs.tolist()
            })

        except Exception as e:
            print(f"Failed for {urls[i]}: {e}")
            results.append({
                'title': f"Error: {urls[i]}",
                'error': str(e)
            })

    return results
'''

# Set the title
st.title("Music Genre Prediction Results")

# Define the table data
data = [
    ["Gloc 9 - Upuan", "hiphop", 0.41, "pop", 0.35],
    ["PALAGI - TJ Monterde", "pop", 0.58, "reggae", 0.10],
    ["HELLMERRY - My Day", "hiphop", 0.58, "pop", 0.09],
    ["SLAPSHOCK - Cariño Brutal", "metal", 0.44, "pop", 0.22],
    ["Awitin Mo At Isasayaw Ko", "disco", 0.46, "reggae", 0.25],
    ["Anak - Freddie Aguilar", "country", 0.33, "blues", 0.15],
    ["With A Smile - Eraserheads", "reggae", 0.62, "hiphop", 0.15],
    ["Your Song - Parokya Ni Edgar", "pop", 0.56, "reggae", 0.13],
    ["Rainbow - South Border", "pop", 0.66, "reggae", 0.16],
    ["Ikaw Lamang - Silent Sanctuary", "pop", 0.53, "disco", 0.09],
    ["Imposible - KZ Tandingan", "pop", 0.65, "hiphop", 0.09],
    ["Florante - Handog", "country", 0.28, "blues", 0.17],
    ["Laguna - Sampaguita", "pop", 0.46, "reggae", 0.11],
    ["Usok", "disco", 0.60, "reggae", 0.11],
    ["Laki Sa Layaw - Mike Hanopol", "disco", 0.49, "rock", 0.15],
    ["Problema - Freddie Aguilar", "pop", 0.43, "disco", 0.18],
    ["Itanong Mo Sa Mga Bata", "disco", 0.43, "country", 0.20],
    ["Himig Natin - Juan Dela Cruz Band", "disco", 0.56, "classical", 0.11],
    ["Ang Buhay Ko", "disco", 0.33, "country", 0.20],
    ["Magsayawan - VST & Co.", "disco", 0.55, "reggae", 0.15],
    ["Sumayaw, Sumunod - The Boyfriends", "pop", 0.55, "rock", 0.13],
    ["Bongga Ka Day - Hotdog", "disco", 0.38, "pop", 0.25],
    ["Annie Batungbakal - Hotdog", "disco", 0.35, "pop", 0.25],
    ["Dalaga - Allmo$t ", "hiphop", 0.69, "reggae", 0.10],
    ["Taguan - Jroa", "hiphop", 0.53, "blues", 0.14],
    ["1096 Gang - PAJAMA PARTY", "hiphop", 0.31, "reggae", 0.30],
    ["Kahit Na Tambay - Fred Engay", "hiphop", 0.64, "blues", 0.08],
    ["Get Low - O SIDE MAFIA X BRGR", "hiphop", 0.82, "reggae", 0.07],
    ["Tell Me Where It Hurts - MYMP", "pop", 0.53, "hiphop", 0.18],
    ["Especially For You - MYMP", "pop", 0.57, "hiphop", 0.18],
    ["OO - Up Dharma Down", "pop", 0.46, "reggae", 0.21],
    ["Weight of the World - LOSTTHREADS", "pop", 0.35, "metal", 0.32],
    ["Kamay Na Bakal", "metal", 0.36, "disco", 0.17],
    ["Gento - SB19", "hiphop", 0.78, "pop", 0.13],
    ["Dilaw - Maki", "pop", 0.58, "disco", 0.10],
    ["Pantropiko - BINI", "pop", 0.96, "reggae", 0.03],
    ["Tala - Sarah Geronimo", "hiphop", 0.33, "rock", 0.24],
    ["Misteryoso - Cup of Joe", "pop", 0.63, "hiphop", 0.22],
    ["Kapayapaan - Tropical Depresion", "pop", 0.69, "hiphop", 0.16],
    ["Estudyante Blues", "disco", 0.71, "pop", 0.09],
    ["Nosi Ba Lasi - Sampaguita", "pop", 0.32, "rock", 0.26],
    ["Banal Na Aso - Yano", "reggae", 0.37, "disco", 0.21],
    ["Kalag sa Gapos - Supremo", "metal", 0.76, "pop", 0.10],
    ["Pangako - Valley of Chrome", "metal", 0.82, "disco", 0.06],
    ["Walang Hanggang Paalam", "blues", 0.34, "pop", 0.26],
    ["Ikaw at Ako - Johnoy Danao", "blues", 0.39, "pop", 0.30],
    ["Harana sa Sarili - Kiyo", "hiphop", 0.53, "reggae", 0.18],
    ["Bakuran - Johnoy Danao", "pop", 0.38, "blues", 0.27],
    ["Philippine Geography - Yoyoy Villame", "disco", 0.23, "country", 0.19],
    ["Kanlungan (Pana-panahon)", "country", 0.49, "pop", 0.18],
    ["Marilag - Dionela", "pop", 0.68, "hiphop", 0.13],
    ["Answer The G - SUPAFLY", "hiphop", 0.70, "pop", 0.17],
    ["BURGIS - Flow G x Hev Abi", "hiphop", 0.62, "pop", 0.28],
    ["Lintik - Brownman Revival", "pop", 0.59, "hiphop", 0.14],
    ["Binibini - Brownman Revival", "reggae", 0.34, "disco", 0.22],
]

# Define column headers
columns = ["Title", "Prediction 1", "Prob 1", "Prediction 2", "Prob 2"]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Show table with sorting
st.dataframe(df, use_container_width=True)

# Set the page title
st.title("Music Genre Prediction vs Actual Result")

# Define the table data
data = [
    ["Gloc 9 - Upuan", "hiphop", "hiphop", "✓"],
    ["PALAGI - TJ Monterde", "pop", "pop", "✓"],
    ["HELLMERRY - My Day", "hiphop", "hiphop", "✓"],
    ["SLAPSHOCK - Cariño Brutal", "metal", "metal", "✓"],
    ["Awitin Mo At Isasayaw Ko - VST & Co.", "disco", "disco", "✓"],
    ["Anak - Freddie Aguilar	", "country", "country", "✓"],
    ["With A Smile - Eraserheads ...", "reggae", "reggae", "✓"],
    ["Your Song - Parokya Ni Edgar", "pop", "pop", "✓"],
    ["Rainbow - South Border", "pop", "pop", "✓"],
    ["Ikaw Lamang - Silent Sanctuary", "pop", "pop", "✓"],
    ["Imposible - KZ Tandingan", "pop", "pop", "✓"],
    ["Florante - Handog", "country", "blues", "✗"],
    ["Laguna - Sampaguita", "pop", "rock", "✗"],
    ["Usok", "disco", "reggae", "✗"],
    ["Laki Sa Layaw - Mike Hanopol", "disco", "rock", "✗"],
    ["Problema - Freddie Aguilar", "pop", "pop", "✓"],
    ["Itanong Mo Sa Mga Bata", "disco", "country", "✗"],
    ["Himig Natin - Juan Dela Cruz Band", "disco", "rock", "✗"],
    ["Ang Buhay Ko", "disco", "country", "✗"],
    ["Magsayawan - VST & Co.", "disco", "disco", "✓"],
    ["Sumayaw, Sumunod - The Boyfriends", "pop", "disco", "✗"],
    ["Bongga Ka Day - Hotdog  ", "disco", "disco", "✓"],
    ["Annie Batungbakal - Hotdog ", "disco", "disco", "✓"],
    ["Dalaga - Allmo$t   ", "hiphop", "hiphop", "✓"],
    ["Taguan - Jroa	", "hiphop", "hiphop", "✓"],
    ["1096 Gang - PAJAMA PARTY ", "hiphop", "hiphop", "✓"],
    ["Kahit Na Tambay - Fred Engay", "hiphop", "hiphop", "✓"],
    ["Get Low - O SIDE MAFIA X BRGR ", "hiphop", "hiphop", "✓"],
    ["Tell Me Where It Hurts - MYMP ", "pop", "pop", "✓"],
    ["Especially For You - MYMP  ", "pop", "pop", "✓"],
    ["OO - Up Dharma Down", "pop", "pop", "✓"],
    ["Weight of the World - LOSTTHREADS", "pop", "metal", "✗"],
    ["Kamay Na Bakal", "metal", "metal", "✓"],
    ["Gento - SB19   ", "hiphop", "pop", "✗"],
    ["Dilaw - Maki  ", "pop", "pop", "✓"],
    ["Pantropiko - BINI ", "pop", "pop", "✓"],
    ["Tala - Sarah Geronimo ", "hiphop", "pop", "✗"],
    ["Misteryoso - Cup of Joe", "pop", "pop", "✓"],
    ["Kapayapaan - Tropical Depression", "pop", "reggae", "✗"],
    ["Estudyante Blues", "disco", "blues", "✗"],
    ["Nosi Ba Lasi - Sampaguita ", "pop", "rock", "✗"],
    ["Banal Na Aso - Yano ", "reggae", "rock", "✗"],
    ["Kalag sa Gapos - Supremo", "metal", "metal", "✓"],
    ["Pangako - Valley of Chrome", "metal", "metal", "✓"],
    ["Walang Hanggang Paalam", "blues", "blues", "✓"],
    ["Ikaw at Ako - Johnoy Danao", "blues", "blues", "✓"],
    ["Harana sa Sarili - Kiyo ", "hiphop", "hiphop", "✓"],
    ["Bakuran - Johnoy Danao", "pop", "blues", "✗"],
    ["Philippine Geography - Yoyoy Villame", "disco", "country", "✗"],
    ["Kanlungan (Pana-panahon)", "country", "country", "✓"],
    ["Marilag - Dionela ", "pop", "pop", "✓"],
    ["Answer The G - SUPAFLY", "hiphop", "hiphop", "✓"],
    ["BURGIS - Flow G x Hev Abi", "hiphop", "hiphop", "✓"],
    ["Lintik - Brownman Revival", "pop", "reggae", "✗"],
    ["Binibini - Brownman Revival ", "reggae", "reggae", "✓"],
]

# Create DataFrame
df = pd.DataFrame(data, columns=["Title", "Prediction", "Actual", "Result"])

# Show DataFrame in Streamlit with full width
st.dataframe(df, use_container_width=True)

st.header("Genre Classifier Evaluation Summary")
st.subheader("Overall Metrics")
st.markdown("""
            - Accuracy: 72.73% — decent for multi-class classification.
            
            - Macro Precision: 78.56% — model predictions are generally correct.
            
            - Macro Recall: 67.08% — some genres are often missed.
            
            - Macro F1-score: 67.91% — moderate balance between precision and recall.
            """)
st.subheader("Per-Genre Performance")
data = [
    ["Blues", "1.00", "0.60", "0.75", "High precision, low recall — model is conservative."],
    ["Country", "0.80", "0.80", "0.80", "Balanced performance."],
    ["Disco", "0.44", "0.80", "0.57", "High recall, low precision — many false positives."],
    ["Hip-hop", "0.82", "0.90", "0.86", "Strong and reliable."],
    ["Metal", "1.00", "0.80", "0.89", "Very accurate."],
    ["Pop", "0.72", "0.87", "0.79", "Good general performance."],
    ["Reggae", "0.50", "0.40", "0.44", "Weak — struggles in both precision and recall."],
    ["Rock", "1.00", "0.20", "0.33", "Only predicts when very confident — misses most actual rock songs."],
]

# Create DataFrame
df = pd.DataFrame(data, columns=["Genre", "Precision", "Recall", "F1-Score", "Notes"])

# Show DataFrame in Streamlit with full width
st.dataframe(df, use_container_width=True)

st.subheader("Key Insights")
st.markdown("""
            - Model performs better on pop and hip-hop
            
            - Conservative for some genres (blues, rock) — high precision, low recall.

            - Disco often overpredicted — low precision.

            - Underperforming genres likely suffer from class imbalance.
            """)
st.subheader("XGB + RFC  Ensemble Classifer on OPM")
st.markdown("""
            - GTZAN is western-centric and OPM might not fit neatly into the considered genres.

            - Many OPM do not adhere to a single genre and incorporate multiple genres.

            - We only considered a 30-second snippet of the songs, which may not represent the entire song.

            - GTZAN is outdated and relatively small and may not capture modern music characteristics.
    
            - No OPM in the training set.

            - Cultural and linguistic context is ignored.
    
            - Classical and jazz is not prevalent in OPM.

            - R&B is an ambiguous genre within this classification, as it is not included in the GTZAN dataset. As a result, R&B songs are often classified as pop rather than blues, despite their distinct musical characteristics.
            """)