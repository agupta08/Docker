import sys
import json
import pickle

import bottle
from bottle import route, run, template, request, post, response
from polyglot.text import Text
import datetime

app = bottle.default_app()
app.config.load_config('./config.ini')


def transliterate_and_split(message):
    words = Text(message).transliterate("en")
    return [word for word in words if len(word) > 1 and word.isalpha()]


def load_model(source):
    with open(source, 'rb') as f:
        return pickle.load(f)

vectoriser = load_model(app.config["myapp.vectorizer_pickle"])
model = load_model(app.config["myapp.model_pickle"])


@post('/api/classify')
def post():
    start_time = datetime.datetime.now()
    #text = request.forms.get('key')
    text = request.json['text']

    X_test = vectoriser.transform([text])

    Y = model.predict(X_test)

    response.content_type = 'application/json'
    data = {
        "class": Y[0].encode('ascii', 'ignore'),
        "response_sec": (datetime.datetime.now()-start_time).total_seconds()
    }
    return json.dumps(data)


@route('/')
def main():
    return template('main')

run(host='0.0.0.0', port = int(sys.argv[1]), debug=True)


