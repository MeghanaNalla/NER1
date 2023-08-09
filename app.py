import spacy
import spacy_streamlit as spt
import streamlit as st

nlp = spacy.load('en_core_web_sm')


def main():
    st.title('Name Entity Recognition App')
    
    menu = ['Home','NER']
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
        
if __name__ == '__main__':
    main()