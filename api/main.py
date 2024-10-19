from fastapi import FastAPI,File,UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
app=FastAPI()

@app.get('/ping')
async def ping():
    return('hi')

def read_file_as_image(data):
    image =np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')  
async def predict(  
   file: UploadFile = File(...)  
):  
    image = read_file_as_image(await file.read()) 
    
if __name__ == '__main__':
    uvicorn.run(app,port=8000,host='localhost')