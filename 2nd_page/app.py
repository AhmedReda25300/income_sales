from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import io

app = Flask(__name__)

FEATURE_SETS = {
    'Net Income': ['Date', 'Total Expenses', 'Total Operating Expenses', 'Zakat/Tax'],
    'Sales': ['Date', 'Gross Income', 'Cost of Sales'],
}

def translate_column_names(df):
    return df.rename(columns={
    'Date': 'Date',
    'المبيعات': 'Sales',
    'تكاليف المبيعات': 'Cost of Sales',
    'إجمالي الدخل': 'Gross Income',
    'اجمالي المصاريف التشغيليه': 'Total Operating Expenses',
    'المصاريف/ الايرادات الأخري': 'Other Expenses/Revenues',
    'إجمالي المصاريف': 'Total Expenses',
    'صافي الدخل قبل الزكاة': 'Net Income Before Zakat',
    'الزكاة/الضريبة': 'Zakat/Tax',
    'صافي الدخل': 'Net Income'
}, inplace=True)

def load_and_preprocess_data(file):
    df = pd.read_excel(file, parse_dates=['Date'])
    translate_column_names(df)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.sort_values('Date')
    return df

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length, -1]  # Target is the last column
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0, validation_split=0.2, callbacks=[early_stopping])
    print(f'\n{"*"*50}\nTraining accuracy: ', history.history['accuracy'][-1])
    print(f'Training loss: ', history.history['loss'][-1])
    return model

def predict_future(model, last_sequence, num_steps):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(num_steps):
        next_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape))
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, -1] = next_pred
    
    return np.array(future_predictions).reshape(-1, 1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = load_and_preprocess_data(file)
            columns = df.columns.tolist()
            return jsonify({"columns": columns, "feature_sets": FEATURE_SETS})
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print('\n\n\n\n\n starting model...')
    file = request.files['file']
    df = load_and_preprocess_data(file)
    
    results = {}
    for target in FEATURE_SETS.keys():
        features = FEATURE_SETS[target]
        data = df[features + [target]].values
        
        data = data[:, 1:]  # Remove the date column for scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        seq_length = 4
        sequences, targets = create_sequences(data_scaled, seq_length)
        train_size = int(len(sequences))
        X_train, y_train = sequences[:train_size], targets[:train_size]
        
        model = build_model((seq_length, data_scaled.shape[1]))
        model = train_model(model, X_train, y_train)
        
        last_sequence = data_scaled[-seq_length:]
        future_predictions_scaled = predict_future(model, last_sequence, 2)
        
        # Prepare dummy array for inverse transform
        dummy = np.zeros((future_predictions_scaled.shape[0], data_scaled.shape[1]))
        dummy[:, -1] = future_predictions_scaled.flatten()
        
        future_predictions = scaler.inverse_transform(dummy)[:, -1]
        
        results[target] = future_predictions.tolist()
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)