import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('E:\\lung_Cancer\\model\\model_lung.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = (prediction[0])
    #output = 0
    if output == 0:
        output = 'No'
    else:
        output = 'YES'

    return render_template('index.html', prediction_text='Is person suffered from Lung Cancer {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
