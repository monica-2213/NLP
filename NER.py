import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the dataset
df = pd.read_csv('menu.csv')  # Replace 'menu.csv' with the path to your dataset
df = df.drop_duplicates(subset='menu_id')  # Remove duplicates
df = df.dropna(subset=['description', 'ingredients'])  # Remove rows with missing values

# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description'] + ' ' + df['ingredients'])

# Function to recommend menu items based on user preferences
def recommend_menu_items(user_preferences, top_n=5):
    # Transform user preferences into a TF-IDF vector
    user_preferences_vector = vectorizer.transform([user_preferences])

    # Calculate cosine similarity between user preferences and all menu items
    similarity_scores = cosine_similarity(user_preferences_vector, tfidf_matrix).flatten()

    # Sort menu items based on similarity scores and get top recommendations
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    recommendations = df.iloc[top_indices]['menu_item']

    return recommendations

# Streamlit app code
def main():
    # App title
    st.title('Menu Recommendation')

    # User input section
    user_preferences = st.text_area('Enter your preferences', height=100)
    if st.button('Recommend'):
        recommendations = recommend_menu_items(user_preferences)
        st.subheader('Recommended Menu Items:')
        for i, item in enumerate(recommendations):
            st.write(f"{i+1}. {item}")

if __name__ == '__main__':
    main()
