import pandas as pd 
import numpy as np 
import streamlit as st 
import plotly.express as px

st.title("Sentiment Analysis of Tweets of US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets of US Airlines")
st.markdown("This application is a dashboard to analyze the sentiments of tweets.")

#Loading data
data_url = r"C:\Users\Pranay\DATA_ML\Tweets.csv"
@st.cache(persist = True)
def load_data():
	data = pd.read_csv(data_url,parse_dates = ['tweet_created']) 
	return data

data = load_data()

# Displaying Random Tweet
st.sidebar.subheader("Show Random Tweet")
random_tweet = st.sidebar.radio('Sentiment',('Positive','Neutral','Negative'))
st.sidebar.markdown(data.query('airline_sentiment == @random_tweet.lower()')[['text']].sample(n=1).iat[0,0])

#Display graphs for no of tweets by sentiment
st.sidebar.subheader("Visualize Number Of Tweets By Sentiment")
select = st.sidebar.selectbox("Visualization Type",['Histogram','Pie Chart'])
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index,'Tweets':sentiment_count.values})

if not st.sidebar.checkbox("Hide",True):
	st.subheader("Number Of Tweets By Sentiment")
	if select == 'Histogram':
		fig = px.bar(sentiment_count,x='Sentiment',y='Tweets',color='Tweets',height=600)
		st.plotly_chart(fig)
	else:
		fig = px.pie(sentiment_count,values = 'Tweets',names = 'Sentiment',color = 'Tweets')
		st.plotly_chart(fig)

#Displaying Location of tweets
st.sidebar.subheader("Time And Place Of Tweets")
hour  = st.sidebar.slider("Hour of Day",0,23)
modified_data = data[data['tweet_created'].dt.hour == hour]
print(data['tweet_created'].dt.hour[:10])
if not st.sidebar.checkbox("Close",True):
	st.subheader("Tweets Location Based On Time Of Day")
	st.markdown("%i Tweets between %i:00 and %i:00"%(len(modified_data),hour,(hour+1)%24))
	st.map(modified_data)
	if st.sidebar.checkbox("Show Raw Data",False):
		st.write(modified_data)


# Displaying breakdown of tweets by airlines
st.sidebar.subheader("Breakdown Of Tweets By Airlines")
choice = st.sidebar.multiselect('Airlines',('US Airways','United','American','Southwest','Delta','Virgin America'))
if len(choice)>0:
	st.subheader("Breakdown Of Sentiments Of Tweets By Different Airlines")
	choice_data = data[data['airline'].isin(choice)]
	fig_choice = px.histogram(choice_data,x='airline',y='airline_sentiment',histfunc='count',color='airline_sentiment',height=600,
		width = 800,facet_col='airline_sentiment',labels={'airline_sentiment':'tweets'})
	st.plotly_chart(fig_choice)




















