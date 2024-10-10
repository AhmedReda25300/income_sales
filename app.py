from flask import Flask, render_template, request, jsonify
import pandas as pd
from flask_cors import CORS
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os
import logging
from sklearn.metrics import r2_score
app = Flask(__name__)
CORS(app)
# Set the directory for uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Group_balance')
def Group_balance():
    return render_template('Group_balance.html')

@app.route('/Form')
def Form():
    return render_template('Form.html')

@app.route('/income_Tax')
def income_Tax():
    return render_template('income_Tax.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        data = pd.read_excel(file_path)  
        #Data Validation
        if not {'Net Income', 'Sales','Zakat/Tax', 'Cost of Sales'}.issubset(data.columns):
            return jsonify({"error": "Missing required columns: Net Income, Sales, Date, Zakat/Tax, and Cost of Sales"}), 400

        if (data[['Net Income', 'Sales', 'Zakat/Tax', 'Cost of Sales']] < 0).any().any():
            return jsonify({"error": "Net Income, Sales, Zakat/Tax, and Cost of Sales must be non-negative"}), 400

        #Data Scaling
        scaler = MinMaxScaler(feature_range=(0, 1)) # to normalize the data 
        scaled_data = scaler.fit_transform(data[['Net Income', 'Sales', 'Zakat/Tax', 'Cost of Sales']])

        # Dataset Creation
        def create_dataset(dataset, time_step=1):
            X, Y = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), :]
                X.append(a)
                Y.append(dataset[i + time_step, :])
            return np.array(X), np.array(Y)

        time_step = 1
        X, y = create_dataset(scaled_data, time_step)

        X = X.reshape(X.shape[0], X.shape[1], 4) 

        # Build and train the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 4)))  # Input shape with 4 features
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(4))  # Predicting all 4 columns
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=150, batch_size=20, verbose=2)

        # After training, get the model's predictions on the training data
        train_predict = model.predict(X)

        # Inverse transform the predictions and true values to original scale
        train_predict_inv = scaler.inverse_transform(train_predict)
        y_inv = scaler.inverse_transform(y)

        last_data = scaled_data[-time_step:].reshape((1, time_step, 4))
        prediction = model.predict(last_data)
        predicted_values = scaler.inverse_transform(prediction)
        print(f'Predicted Net Income: {predicted_values[0][0], predicted_values[0][1], predicted_values[0][2], predicted_values[0][3]}')
        # Return the predictions and R² scores
        return jsonify({
            'predictedNetIncome': int(predicted_values[0][0]),
            'predictedSales': int(predicted_values[0][1]),
            'predictedZakat/Tax': int(predicted_values[0][2]),
            'predictedCost_of_Sales': int(predicted_values[0][3]),
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        logging.error(f"Error processing the uploaded file: {e}")
        return jsonify({"error": "Error processing the file"}), 500

if __name__ == '__main__':
    app.run(debug=True)