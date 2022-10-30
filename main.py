from flask import Flask, render_template, request
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# import flask
import pickle
app = Flask(__name__)

#model Export
with open('model_pickle', 'rb') as f:
    model = pickle.load(f)

#TFIDF extraction
with open('transform_cv', 'rb') as t:
     cv = pickle.load(t)

@app.route('/')
def home():
     return render_template('index.html')

@app.route('/predict', methods=['post', 'get'])
def submit():
     text = "Hello"
     if request.method=='POST':
         text = str(request.form['txt'])
         wordnet = WordNetLemmatizer()
         corpus = []
         review = re.sub('[^a-zA-Z]', ' ', text)
         review = review.lower()
         review = review.split()
         review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
         review = ' '.join(review)
         corpus.append(review)

         vect = cv.transform(corpus).toarray()
         ans = model.predict(vect)
         res="Not Spam"
         if ans[0] == 1: res = "Spam"
         return render_template('predict.html', result=res, data=text)

if __name__ == '__main__':
     app.run(debug=True)

