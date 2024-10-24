import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Carregar o modelo
model = load_model('parkinson_mri_cnn_model_3.h5')

# Diretório das imagens de teste
test_dir = 'imagens/parkinsons_dataset/test'

# Função para carregar e pré-processar uma imagem
def load_and_preprocess_image(image_path):
    with open(image_path, 'rb') as file:
        img = image.load_img(io.BytesIO(file.read()), target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Listas para armazenar as predições e rótulos verdadeiros
y_true = []
y_pred = []

# Processar imagens e fazer predições
for class_name in ['normal', 'parkinson']:
    class_dir = os.path.join(test_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = load_and_preprocess_image(img_path)
        
        prediction = model.predict(img)
        predicted_prob = float(prediction[0][1])  # Probabilidade de Parkinson
        predicted_class = 1 if predicted_prob > 0.5 else 0
        
        y_true.append(0 if class_name == 'normal' else 1)
        y_pred.append(predicted_class)

# Calcular métricas
conf_matrix = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Criar a figura e os subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})

# Plotar matriz de confusão
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Matriz de Confusão')
ax1.set_xlabel('Predito')
ax1.set_ylabel('Verdadeiro')

# Adicionar métricas abaixo da matriz de confusão
metrics_text = f"""
Precisão: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}
Acurácia: {accuracy:.4f}
"""
ax2.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12)
ax2.axis('off')

# Ajustar o layout e salvar a figura
plt.tight_layout()
plt.savefig('confusion_matrix_with_metrics.png')
plt.close()

print("Matriz de confusão com métricas salva como 'confusion_matrix_with_metrics.png'")

# Imprimir métricas no console também
print(metrics_text)
