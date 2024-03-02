import os
import numpy as np
import joblib
from collections import Counter
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import preproc as pr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# models saved by the training scripts
theta_logistic = joblib.load(os.path.join(BASE_DIR, 'logisticregression_manual_weights.joblib'))
knn_classifier = joblib.load(os.path.join(BASE_DIR, 'knn_model.joblib'))
nn_weights = joblib.load(os.path.join(BASE_DIR, 'neural_network_weights.joblib'))
nb_model_params_with_pca = joblib.load(os.path.join(BASE_DIR, 'naive_bayes_model_with_pca.joblib'))

class_mapping = {
    0: 'Benign',
    1: 'Malignant',
    2: 'Normal'
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(inputs, weights):
    layer_outputs = [inputs]
    for w in weights[:-1]:
        layer_inputs = np.dot(layer_outputs[-1], w)
        layer_outputs.append(sigmoid(layer_inputs))

    output_inputs = np.dot(layer_outputs[-1], weights[-1])
    return layer_outputs, output_inputs

def predict_nn(inputs, weights):
    _, output_inputs = forward_pass(inputs, weights)
    output = np.exp(output_inputs - np.max(output_inputs, axis=1, keepdims=True))
    output /= np.sum(output, axis=1, keepdims=True)
    return np.argmax(output, axis=1)

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def predict_logistic_regression(image, theta_logistic):
    image_bias = np.insert(image, 0, 1)  # add bias
    prediction_prob = softmax(np.dot(image_bias, theta_logistic))
    prediction = np.argmax(prediction_prob)

    return class_mapping[prediction]

def predict_naive_bayes_internal(model_params, X):
    predictions = []
    for x in X:
        class_scores = []
        for c in model_params['classes']:
            class_prob = np.log(model_params['class_probs'][c])
            for i, feature_value in enumerate(x):
                mean = model_params['feature_probs'][c]['mean'][i]
                std = model_params['feature_probs'][c]['std'][i]
                log_likelihood = -0.5 * np.log(2 * np.pi * std**2) - ((feature_value - mean)**2) / (2 * std**2)
                class_prob += log_likelihood

            class_scores.append(class_prob)

        predicted_class = model_params['classes'][np.argmax(class_scores)]
        predictions.append(predicted_class)

    return np.array(predictions)

def predict_naive_bayes_pca(model_params, pca, X):
    X_pca = pca.transform(X)
    predictions = predict_naive_bayes_internal(model_params['nb_model_params'], X_pca)
    class_names = [class_mapping[num] for num in predictions]

    return class_names

def predict_knn_internal(sample, knn_model_params):
    distances = np.linalg.norm(knn_model_params['X_train'] - sample, axis=1)
    nearest_indices = np.argsort(distances)[:knn_model_params['k']]
    nearest_labels = knn_model_params['y_train'][nearest_indices]

    most_common = Counter(nearest_labels).most_common(1)
    predicted_label = most_common[0][0]

    return predicted_label

def predict_knn(sample, knn_model_params):
    predicted_label = predict_knn_internal(sample, knn_model_params)
    return class_mapping[predicted_label]

def predict_neural_network(image, weights):
    prediction = predict_nn(np.array([image]), weights)
    return class_mapping[prediction[0]]

# same preprocessing used for training
def preprocess_image(image_path):
    return pr.extract_features(pr.load_image(image_path))

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

        preprocessed_image = preprocess_image(filepath)

        selected_models = request.form.getlist('models')

        results = {}

        for selected_model in selected_models:
            if selected_model == 'Logistic Regression':
                results[selected_model] = predict_logistic_regression(preprocessed_image, theta_logistic)
            elif selected_model == 'Naive Bayes':
                results[selected_model] = predict_naive_bayes_pca(nb_model_params_with_pca, nb_model_params_with_pca['pca'], [preprocessed_image])[0]
            elif selected_model == 'KNN':
                results[selected_model] = predict_knn(preprocessed_image, knn_classifier)
            elif selected_model == 'Neural Network':
                results[selected_model] = predict_neural_network(preprocessed_image, nn_weights)
            else:
                return render_template('index.html', error='Invalid model selected')

        return render_template('index.html', results=results, image=filename)

    else:
        return render_template('index.html', error='File format not supported')

if __name__ == '__main__':
    app.run(debug=True)
