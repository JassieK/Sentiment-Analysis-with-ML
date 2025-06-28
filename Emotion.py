import joblib
import gradio as gr
import nbimporter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import neattext as nt 
import neattext.functions as nfx

model = joblib.load("emotion_sgd_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
encoder = joblib.load("label_encoder.joblib")

lemmatizer = WordNetLemmatizer()

spell = SpellChecker()
def correct_it(tokens):
    return [spell.correction(word) if word not in spell else word for word in tokens]

def text_preprocess(text):
    text = text.lower()
    
    text = nfx.remove_userhandles(text)
    text = nfx.remove_stopwords(text)
    text = nfx.remove_punctuations(text)
    
    words = word_tokenize(text)
    
    words = correct_it(words)
    
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word is not None]
    
    text_processed = " ".join(lemmatized_words)
    
    return text_processed

def predict_emotion(text):
    clean_text = text_preprocess(text)
    vector = vectorizer.transform([clean_text])
    pred = model.predict(vector)
    label = encoder.inverse_transform(pred)[0]
    return label

gui = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Type a sentence here..."),
    outputs="text",
    title="Emotion Detector",
    description="Enter a sentence and find out the predicted emotion!"
)

if __name__ == "__main__":
    gui.launch()
    gui.launch(inbrowser=True)

