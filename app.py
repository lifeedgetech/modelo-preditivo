from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Carregar a chave API
def load_api_key():
    with open('api_key.txt', 'r') as f:
        return f.read().strip()

api_key = load_api_key()

# Carregar o modelo treinado
model = load_model('parkinson_mri_cnn_model.h5')
model.summary()

# Rota para lidar com o upload da imagem e retornar a predição
@app.route('/predict', methods=['POST'])
def predict():
    # Verificar a chave API para autenticação
    request_api_key = request.headers.get('Authorization')
    if request_api_key != api_key:
        return jsonify({'error': 'Acesso não autorizado'}), 401
   
    # Verificar se foi enviada uma imagem
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400
   
    # Ler a imagem enviada
    file = request.files['file']
    img = image.load_img(file, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
   
    # Fazer a predição
    prediction = model.predict(img)
   
    # Decodificar a predição
    if prediction[0][0] > 0.5:
        result = 'Parkinson detectado'
    else:
        result = 'Normal'
   
    return jsonify({'result': result}), 200

if __name__ == '__main__':
    app.run(debug=True)
