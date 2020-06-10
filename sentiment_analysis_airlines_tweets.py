import pandas as pd 
import numpy as np
import re
import nltk
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve,precision_score,recall_score,f1_score
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

#Loading Data
data_url = '/content/Tweets.csv'
df = pd.read_csv(data_url)
np.set_printoptions(precision=2)

#Preprocessing Data
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub(r'[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text
df['text'] = df['text'].apply(preprocessor)
label = LabelEncoder()
df['airline_sentiment'] = label.fit_transform(df['airline_sentiment'])

#Tokenizing
porter = PorterStemmer()
stop = stopwords.words('english')
def tokenizer_porter(text):
	return [porter.stem(word) for word in text.split() if word not in stop ]

#Transform text data into TF-IDF Vectors
tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,
                        tokenizer=tokenizer_porter,use_idf=True,norm='l2',smooth_idf=True)
y = df.airline_sentiment.values
x = tfidf.fit_transform(df.text)

#Logistic Regression
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,test_size=0.2,shuffle=False)
model = LogisticRegressionCV(Cs=20,cv=5,random_state=0,multi_class='multinomial',n_jobs=-1,verbose=3,max_iter=300)
model.fit(x_train,y_train)
saved_model_tweets=open('saved_model_tweets.sav','wb')
pickle.dump(model,saved_model_tweets)
saved_model_tweets.close()

#Load trained model
filename = '/content/saved_model_tweets.sav'
saved_clf = pickle.load(open(filename,'rb'))

#Results
accuracy = saved_clf.score(x_test,y_test).round(3)
y_pred = saved_clf.predict(x_test)
print("Accuracy: ",accuracy)

#Confusion Matrix
plot_confusion_matrix(saved_clf,x_test,y_test)