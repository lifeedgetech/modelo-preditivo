from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = FastAPI()

# Carregar o modelo treinado
model = load_model('parkinson_mri_cnn_model.h5', custom_objects={'loss': CustomLoss(reduction='none')})


@app.post("/predict")
async def predict(file: UploadFile):
    # Verificar se foi enviada uma imagem
    if not file:
        raise HTTPException(status_code=400, detail="Nenhuma imagem enviada")
    
    # Ler a imagem enviada
    contents = await file.read()
    img = image.load_img(io.BytesIO(contents), target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    
    # Fazer a predição
    prediction = model.predict(img)
    
    
    return JSONResponse(content={"result": prediction[0][0]}, status_code=200)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
