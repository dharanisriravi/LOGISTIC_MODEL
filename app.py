from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load vectorizer and model (ORDER IS IMPORTANT)
with open('fake_news.pkl', 'rb') as f:
    tfidf, model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        news = request.form['news']
        if news.strip():
            vec = tfidf.transform([news])
            pred = model.predict(vec)[0]
            prediction = "🔴 FAKE News" if pred == 1 else "🟢 REAL News"
        else:
            prediction = "Please enter news content!"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
