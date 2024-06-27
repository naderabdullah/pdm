import os
import json
import pandas as pd

def json_to_csv(input_dir, output_csv):
    json_files = [pos_json for pos_json in os.listdir(input_dir) if pos_json.endswith('.json')]

    # Initialize headers
    headers = set()
    for file_name in json_files:
        with open(os.path.join(input_dir, file_name), 'r') as f:
            data = json.load(f)
            sensor_values = data.get('sensorValues', {})
            for key in sensor_values:
                headers.add(key)
    
    headers = sorted(headers)
    headers.append('File')
    headers.append('Timestamp')  # Add the timestamp to the headers

    all_rows = []

    for file_name in json_files:
        with open(os.path.join(input_dir, file_name), 'r') as f:
            data = json.load(f)
            sensor_values = data.get('sensorValues', {})
            
            # Collect all unique timestamps from the sensor values
            timestamps = set()
            for key in sensor_values:
                for entry in sensor_values[key]:
                    timestamps.add(entry.get('timestamp'))

            # Create rows for each unique timestamp
            for timestamp in sorted(timestamps):
                row = {'File': file_name, 'Timestamp': timestamp}
                for key in sensor_values:
                    value_found = False
                    for entry in sensor_values[key]:
                        if entry.get('timestamp') == timestamp:
                            row[key] = entry.get('value', None)
                            value_found = True
                            break
                    if not value_found:
                        row[key] = None  # If no value for this timestamp, set to None
                all_rows.append(row)

    df = pd.DataFrame(all_rows)  # Create DataFrame from all rows
    print(f"Initial number of rows: {len(df)}")
    print("First few rows before dropping NaNs:")
    print(df.head())

    # Fill NaN values with a default value or drop columns with all NaNs
    df.dropna(axis=1, how='all', inplace=True)

    # Sort the DataFrame by 'Timestamp'
    #df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')  # Convert 'Timestamp' to datetime

    df = df.sort_values(by='Timestamp')#.reset_index(drop=True)

    # Verify sorting
    print("First few rows after sorting by Timestamp:")
    print(df.head())

    print(f"Number of rows after handling NaNs: {len(df)}")

    df.to_csv(output_csv, index=False)

input_directory = r'records'
output_csv_file = r'output_fake.csv'

json_to_csv(input_directory, output_csv_file)

