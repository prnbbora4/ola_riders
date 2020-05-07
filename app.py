from flask import Flask, request, render_template, url_for
import pickle
import numpy as np
import math


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Predict', methods=['POST'])
def predict():
    init_features = [int(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', p_text="number of weekly riders should be {}".format(math.floor(output)))


if __name__ == '__main__':
    app.run(debug=True)
