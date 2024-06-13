import json
import os
import random
import shutil

# Create list of available sensors
def create_sensor_list(files, path):
    with open(os.path.join(path, files[0])) as f:
        data = json.loads(f.read())
    sensors = [i for i in data.get('sensorValues', {})]
    return sensors

# Extract values from sensors and create dict
def extract_data(file, sensors, path):
    dicts = {}
    with open(os.path.join(path, file), "r") as f:
        data = json.loads(f.read())

    dicts['File'] = file

    if 'sensorValues' in data:
        for i in sensors:
            if i in data['sensorValues'] and data['sensorValues'][i]:
                dicts[i] = data['sensorValues'][i][0]['value']
    
    return dicts

# Copy files and remove timestamps
def copy_records(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count = 1

    for file_name in os.listdir(input_dir):
        if count == 20001:
            break

        if file_name.endswith('.json'):
            with open(os.path.join(input_dir, file_name), 'r') as f:
                data = json.loads(f.read())

            # Check Probe_Temp range
            """
            if 'sensorValues' in data and 'Probe_Temp' in data['sensorValues'] and data['sensorValues']['Probe_Temp']:
                probe_temp_value = float(data['sensorValues']['Probe_Temp'][0]['value'])
                if probe_temp_value < 4 or probe_temp_value > 20:
                    continue  # Skip this record
            else:
                continue  # Skip if Probe_Temp is missing or empty
            """
            # Remove 'timestamp' keys
            if 'machineUDID' in data:
                del data['machineUDID']

            if 'sensorValues' in data:
                if 'Motion' in data['sensorValues']:
                    del data['sensorValues']['Motion']
                for sensor in data['sensorValues']:
                    for entry in data['sensorValues'][sensor]:
                        if 'timestamp' in entry:
                            del entry['timestamp']
            
            with open(os.path.join(output_dir, f'File_{count}.json'), 'w') as f:
                json.dump(data, f, indent=4)
                count += 1

def generate_bad_records(original_dicts, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(200):
        new_record = {'sensorValues': {}}
        for key in original_dicts[0].keys():
            if key == 'File':
                new_record[key] = f'File_{i+1}.json'
            elif key == 'Probe_Temp':
                min_val = 20  # Generate a bad value outside the acceptable range
                max_val = 30
                new_record['sensorValues'][key] = [{"value": str(round(random.uniform(min_val, max_val), 2))}]
            else:
                original_values = [float(d[key]) for d in original_dicts if key in d and d[key] not in ['', 'False', 'True']]
                if original_values:
                    min_val = min(original_values)
                    max_val = max(original_values)
                    new_record['sensorValues'][key] = [{"value": str(round(random.uniform(min_val, max_val), 2))}]
    
        with open(os.path.join(output_dir, f'File_{i+1}.json'), 'w') as f:
            json.dump(new_record, f, indent=4)

def main():
    # Make list of .json files
    path = r'/home/pepsico/pepsi/hg-portal-messages/_data'
    files = os.listdir(path)
    dicts = []

    sensors = create_sensor_list(files, path)

    # Extract data from files
    count = 0
    for file in files:
        if file.endswith('.json'):
            extracted_data = extract_data(file, sensors, path)
            if extracted_data:  # Ensure the extracted data is not empty
                dicts.append(extracted_data)
                count += 1
    
    output_dir = 'records'
    copy_records(path, output_dir)
    generate_bad_records(dicts, output_dir)

if __name__ == '__main__':
    main()






