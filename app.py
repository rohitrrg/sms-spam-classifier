from flask import Flask, app, render_template, request
import re
from nltk.stem.porter import PorterStemmer
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
word2vec = pickle.load(open('word2vec.pkl', 'rb'))
ps = PorterStemmer()
stwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


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

    review = [ps.stem(word) for word in review if not word in stwords]
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