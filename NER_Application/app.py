import spacy
import spacy_streamlit as spt
import streamlit as st
from transformers import pipeline


nlp = spacy.load('en_core_web_md')
summarizer = pipeline('summarization')
sentiment = pipeline("sentiment-analysis", model='cardiffnlp/twitter-roberta-base-sentiment-latest', tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")

def main():
    st.title('Name Entity Recognition App')
    
    menu = ['Home','NER', 'Sentiment analysis', 'summarizer']
    choice = st.sidebar.selectbox('menu', menu)
    if choice == 'Home':
        st.subheader('Word Tokenizer')
        raw_text= st.text_area('Text To Tokenize','Enter Text Here')
        docs= nlp(raw_text)
        if st.button('Tokenize'):
            spt.visualize_tokens(docs)
    elif choice == 'NER':
        st.subheader('Name Entity Recognition')
        raw_text= st.text_area('Text To Tokenize','Enter Text Here')
        docs= nlp(raw_text)
        if st.button('Tokenize'):
            spt.visualize_ner(docs)
            spt.visualize_parser(docs)
    elif choice == 'Sentiment analysis':
        st.subheader('Sentiment analyser')
        raw_text= st.text_area('Text To analyze','Enter Text Here')
        docs= nlp(raw_text)
        if st.button('analyse'):
            st.success(sentiment(raw_text))
    elif choice == 'summarizer':
        st.subheader('Text summarizer')
        raw_text= st.text_area('Text To summarize','Enter Text Here')
        docs= nlp(raw_text)
        # summary_options = st.selectbox("choice you summarizer",("gensim", "sumy"))
        if st.button('summarize'):
            st.success(summarizer(raw_text))
            
            
if __name__ == '__main__':
    main()