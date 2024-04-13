from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pickled model
with open('model_ensemble.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_values = [float(request.form[feature]) for feature in request.form]

    # Make prediction
    prediction = model.predict([input_values])

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
