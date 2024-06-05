from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    files = os.listdir('generated_records')
    return render_template('index.html', files=files)

@app.route('/generated_records/<filename>')
def get_file(filename):
    return send_from_directory('generated_records', filename)

if __name__ == '__main__':
    app.run(debug=True)
    