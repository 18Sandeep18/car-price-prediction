from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load model and scaler
with open('codes/model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('codes/model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        salary = float(request.form['salary'])

        # Preprocess
        input_data = [[age, salary]]
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)

        if prediction[0] == 1:
            result = "Customer is interested to purchase the car 😊"
        else:
            result = "Customer is not interested to purchase a car 😔"

        return render_template('result.html', prediction=result)

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/chat-predict', methods=['POST'])
def chat_predict():
    data = request.json
    age = float(data['age'])
    salary = float(data['salary'])

    input_data = [[age, salary]]
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        result = "Customer is interested to purchase the car 😊"
    else:
        result = "Customer is not interested to purchase a car 😔"

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
