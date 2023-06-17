from textblob import TextBlob
import streamlit as st 
import pandas as pd
import cleantext
from googletrans import Translator
st.header('Mutilingiual Sentiment Analysis')
with st.expander('Translate And Analysis Sentiment'):
    translator = Translator()
    def analyze(x):
        if(x>=0.5):
            return "Postive"
        elif x<=-0.5:
            return "Negative"
        else:
            return "Neutral" 
        

    trans = st.text_input('Translate in your Language:')
    if trans:
        out = translator.translate(trans,dest="en")
        st.write(out.text)
        text = out.text
        blob = TextBlob(text)
        st.write('Polarity: ',round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ',round(blob.sentiment.subjectivity,2))
        st.write('Sentiment: ',analyze(round(blob.sentiment.polarity,2)))



with st.expander('Analyze Text'):
    text=st.text_input('Text here :')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ',round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ',round(blob.sentiment.subjectivity,2))
        st.write('Sentiment: ',analyze(round(blob.sentiment.polarity,2)))
    pre = st.text_input('Clean Text: ')
    if pre:
       st.write(cleantext.clean(pre,clean_all=False,extra_spaces=True,stopwords=True,lowercase=True,numbers=True,punct=True)) 
    

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity
    
    def analyze(x):
        if(x>=0.5):
            return "Postive"
        elif x<=-0.5:
            return "Negative"
        else:
            return "Neutral" 

    if upl:
        df=pd.read_csv(upl)
        df['score']=df['tweet'].apply(score)
        df['analysis']=df['score'].apply(analyze)
        st.write(df.head(10))

        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data = csv,
            file_name="Sentiment Analysis.csv",
            mime='text/csv'
        )        