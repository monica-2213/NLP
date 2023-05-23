import streamlit as st
import pandas as pd
import spacy

# Load the dataset
data = pd.read_csv('food_nutrition_dataset.csv')

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

# Function to detect allergens in the ingredient list using NER
def detect_allergens_ner(ingredient_list):
    doc = nlp(ingredient_list)
    allergens = []
    for ent in doc.ents:
        if ent.label_ == 'ALLERGEN':
            allergens.append(ent.text)
    return allergens

# Function to detect allergens in the ingredient list using word embeddings
def detect_allergens_embeddings(ingredient_list, allergens):
    detected_allergens = []
    for allergen in allergens:
        if allergen.lower() in ingredient_list.lower():
            detected_allergens.append(allergen)
    return detected_allergens

# Streamlit app
def main():
    st.title("Allergen Detection App")
    st.write("Enter a list of ingredients and check for potential allergens.")

    # Get user input
    ingredient_list = st.text_area("Enter the list of ingredients (separated by commas)")

    if st.button("Detect Allergens (NER)"):
        detected_allergens_ner = detect_allergens_ner(ingredient_list)
        if detected_allergens_ner:
            st.warning("Potential allergens detected (NER):")
            for allergen in detected_allergens_ner:
                st.write("- " + allergen)
        else:
            st.success("No potential allergens detected using NER.")

    if st.button("Detect Allergens (Word Embeddings)"):
        allergens = ['peanuts', 'milk', 'eggs', 'soy', 'wheat', 'tree nuts', 'fish', 'shellfish']
        detected_allergens_embeddings = detect_allergens_embeddings(ingredient_list, allergens)
        if detected_allergens_embeddings:
            st.warning("Potential allergens detected (Word Embeddings):")
            for allergen in detected_allergens_embeddings:
                st.write("- " + allergen)
        else:
            st.success("No potential allergens detected using Word Embeddings.")

if __name__ == '__main__':
    main()
