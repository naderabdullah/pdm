import json
import os
import random

# create list of available sensors
def create_sensor_list(files, path):
    f = open(path + '/' + files[0])
    data = json.loads(f.read())
    sensors = []

    for i in data['sensorValues']:
        sensors.append(i)
        #print(i)

    f.close()       
    return sensors

# extract values from sensors and create dict
def extract_data(file, sensors, path):
    dicts = {}
    f = open(path + '/' + file, "r")

    data = json.loads(f.read())

    dicts['File'] = file
    motion = []

    for i in sensors:
        if i == 'Motion':
            for j in range(59):
                motion.append(data['sensorValues'][i][j]['value'])
            dicts[i] = motion
        elif data['sensorValues'][i]:
            dicts[i] = data['sensorValues'][i][0]['value']

    f.close()  
    return dicts

# generating records for training model
def generate_records(original_dicts, output_dir, num_records=10000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(num_records):
        new_record = {}
        for key in original_dicts[0].keys():
            if key == 'File':
                new_record[key] = f'GeneratedFile_{i+1}.json'
            elif key == 'Motion':
                new_record[key] = random.choice(original_dicts)['Motion']
            else:
                original_values = [float(d[key]) for d in original_dicts if key in d and d[key] != '']
                if original_values:
                    min_val = min(original_values)
                    max_val = max(original_values)
                    new_record[key] = str(round(random.uniform(min_val, max_val), 2))
    
        with open(os.path.join(output_dir, f'GeneratedFile_{i+1}.json'), 'w') as f:
            json.dump(new_record, f, indent=4)

def main():
    # make list of .json files
    path = r'/home/pepsico/pepsi/preset_records'
    files = os.listdir(path)
    dicts = []

    sensors = create_sensor_list(files, path)

    # extract data from files
    count = 0
    for file in files:
        if file.endswith('.json'):
            dicts.append(extract_data(file, sensors, path))
            #print(dicts[count], '\n')
            count+=1
    
    output_file = 'generated_records'
    generate_records(dicts, output_file)

    #print(vals)
    