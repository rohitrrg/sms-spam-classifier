from flask import Flask, app, render_template, request
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
word2vec = pickle.load(open('word2vec.pkl', 'rb'))
ps = PorterStemmer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    input_txt = request.form.get("sms")
    review = re.sub('[^a-zA-Z]', ' ', input_txt)
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    input_vec = word2vec.transform([review])
    pred = model.predict(input_vec)

    if pred[0] == 0:
        output = 'Ham'
    else:
        output = 'Spam'

    return render_template('index.html', 
                            sms = input_txt,
                             prediction_text='prediction : \n{}'.format(output))


if __name__=="__main__":
    app.run(port="8000", debug=True)