from fastapi import FastAPI, UploadFile, HTTPException

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="Nenhuma imagem enviada")

    print("Imagem recebida com sucesso")

    return {"message": "ok"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
