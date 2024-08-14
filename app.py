from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)
