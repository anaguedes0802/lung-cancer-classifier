import os
import joblib
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import preproc as pr
from features import FeatureExtractor
import logistic_regression_manual
import knn_manual
import naive_bayes_manual
import ann_manual

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

class_mapping = {
    0: 'Benign',
    1: 'Malignant',
    2: 'Normal'
}

# model name in the page -> (saved model, predict function)
models = {
    'Logistic Regression': (joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.joblib')), logistic_regression_manual.predict_labels),
    'KNN': (joblib.load(os.path.join(MODELS_DIR, 'knn.joblib')), knn_manual.predict_labels),
    'Naive Bayes': (joblib.load(os.path.join(MODELS_DIR, 'naive_bayes.joblib')), naive_bayes_manual.predict_labels),
    'Neural Network': (joblib.load(os.path.join(MODELS_DIR, 'neural_network.joblib')), ann_manual.predict_labels),
}

extractor = FeatureExtractor()

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(ROOT_DIR, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file sent')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No file selected')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # same preprocessing and features used for training
        features = extractor.transform([pr.load_image(filepath)])

        results = {}
        for selected_model in request.form.getlist('models'):
            if selected_model not in models:
                return render_template('index.html', error='Invalid model selected')
            model, predict_fn = models[selected_model]
            results[selected_model] = class_mapping[int(predict_fn(model, features)[0])]

        return render_template('index.html', results=results, image=filename)

    else:
        return render_template('index.html', error='File format not supported')

if __name__ == '__main__':
    app.run(debug=True)
