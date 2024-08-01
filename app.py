from flask import Flask, jsonify, render_template, request, send_from_directory
from datetime import datetime, timedelta
import pandas as pd
from data_module import update_combined_data, add_new_data, normalize_data, load_components, save_component, expand_model, detect_anomalies, predict_faulty_component, prepare_training_data
from sensor_module import get_sensor_data, initialize_sensors
from model_module import load_trained_model, train_model

app = Flask(__name__)

# Initialize global variables
model = load_trained_model()
df_combined = pd.read_csv('new_data.csv')
start_timestamp = datetime.now()
scaler = None
alert1 = alert2 = alert3 = alert4 = False

@app.route('/')
def index():
    update_combined_data()
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/data')
def data():
    global df_combined, model, alert1, alert2, alert3, alert4, start_timestamp

    df_combined = pd.read_csv('new_data.csv')
    add_new_data(start_timestamp)
    start_timestamp = start_timestamp + timedelta(minutes=1)

    component_labels = load_components()
    timestamps_combined, data_combined = get_sensor_data(df_combined)
    
    with open('threshold.txt', 'r') as f:
        temp_threshold = float(f.read())
    board_voltage_threshold = 4.65
    voltage_threshold = 115

    anomaly_data = detect_anomalies(
        data_combined, timestamps_combined, temp_threshold, board_voltage_threshold, voltage_threshold
    )
    faulty_component, accuracy, alert1, alert2, alert3, alert4 = predict_faulty_component(
        data_combined, component_labels, model, temp_threshold, board_voltage_threshold, voltage_threshold
    )

    return jsonify({
        'timestamps': timestamps_combined.tolist(),
        'y_temp': anomaly_data['temp_values'].tolist(),
        'y_board_voltage': anomaly_data['board_voltage_values'].tolist(),
        'y_voltage': anomaly_data['voltage_values'].tolist(),
        'temp_anomaly_timestamps': anomaly_data['temp_anomaly_timestamps'].tolist(),
        'board_voltage_anomaly_timestamps': anomaly_data['board_voltage_anomaly_timestamps'].tolist(),
        'voltage_anomaly_timestamps': anomaly_data['voltage_anomaly_timestamps'].tolist(),
        'y_temp_highs': anomaly_data['temp_values'][anomaly_data['temp_highs']].tolist(),
        'y_board_voltage_lows': anomaly_data['board_voltage_values'][anomaly_data['board_voltage_lows']].tolist(),
        'y_voltage_lows': anomaly_data['voltage_values'][anomaly_data['voltage_lows']].tolist(),
        'sustained_anomaly_timestamps': anomaly_data['sustained_anomaly_timestamps'].tolist(),
        'y_sustained_anomalies': anomaly_data['temp_values'][anomaly_data['sustained_anomalies']].tolist(),
        'temp_threshold': temp_threshold,
        'board_voltage_threshold': board_voltage_threshold,
        'voltage_threshold': voltage_threshold,
        'faulty_component': faulty_component,
        'accuracy': accuracy,
        'alert1': alert1,
        'alert2': alert2,
        'alert3': alert3,
        'alert4': alert4
    })

@app.route('/get_components')
def get_components():
    update_combined_data()
    components = load_components()
    return jsonify({'components': components})

@app.route('/submit_form')
def submit_form():
    update_combined_data()
    return render_template('submit.html')

@app.route('/add_component', methods=['POST'])
def add_component():
    data = request.get_json()
    new_component = data.get('new_component', '').strip().lower()
    
    if new_component:
        component_labels = load_components()
        if new_component not in component_labels:
            save_component(new_component)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'exists'})
    return jsonify({'status': 'error'})

@app.route('/submit', methods=['POST'])
def submit():
    global model, df_combined
    data = request.get_json()
    repair_time = data['repair_time']
    time_delta = int(data['time_delta'])
    components = data['components']
    new_component = data.get('new_component', '').strip().lower()

    component_labels = load_components()
    if new_component and new_component not in component_labels:
        component_labels.append(new_component)
        save_component(new_component)

    repair_time = datetime.strptime(repair_time, '%m/%d/%Y %H')
    df_combined['Timestamp'] = pd.to_datetime(df_combined['Timestamp'])

    anomalies_before_repair = df_combined[(df_combined['Timestamp'] <= repair_time) &
                                          (repair_time - df_combined['Timestamp'] <= timedelta(hours=time_delta))]
    if not anomalies_before_repair.empty:
        X_train, y_train = prepare_training_data(anomalies_before_repair, component_labels, new_component, components)
        
        print(y_train)

        if model.output_shape[1] != len(component_labels):
            model = expand_model(model, len(component_labels))
        
        train_model(model, X_train, y_train)
        model.save('LSTM.keras')

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
