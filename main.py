import json
import secrets
import base64

import os
import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from loguru import logger
from starlette.responses import JSONResponse
from configs.test_config import TestConfiguration
import test
import shutil

from fastapi.middleware.cors import CORSMiddleware
from utils.metrics import metric_ssim


app_title = "T1 Generator T2 API"
app_version = "0.0.1"
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_fake_id_card(image):
    pass

async def get_image_file(file):
    extension = file.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")
    if not extension:
        e = "Image must be jpg/jpeg/png format!"
        raise HTTPException(status_code=400, detail=e)

    try:
        contents = await file.read()
        image = cv2.imdecode(np.asarray(bytearray(contents), dtype=np.uint8), 1)
        return image
    except Exception as e:
        logger.debug(e)
        raise HTTPException(status_code=500, detail=e)


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return JSONResponse(get_openapi(title=app_title, version=app_version, routes=app.routes))


@app.get("/docs", include_in_schema=False)
async def get_documentation():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


async def preprocess_image_file(file):
    tumor_image = await get_image_file(file)
    # tumor_image = cv2.imread(file)
    if len(tumor_image.shape) != 3:
        raise HTTPException(
            status_code=400, detail="Selfie image should not be grayscale image"
        )

    if not os.path.isdir('./datasets/evaluating/test/'):
        os.makedirs('./datasets/evaluating/test/')
    else:
        shutil.rmtree('./datasets/evaluating/test/')
        os.makedirs('./datasets/evaluating/test/')

    # id_card_image = cv2.cvtColor(id_card_image, cv2.COLOR_BGR2GRAY)
    id_card_image=cv2.resize(tumor_image,(256,256))
    for i in range(10):
        save_path = '{}img_test_{}.jpg'.format('./datasets/evaluating/test/', i)
        canvas = np.zeros((256, 256 * 2, 3))
        canvas[:, :256] = id_card_image
        canvas[:, 256:] = id_card_image
        cv2.imwrite(save_path, canvas)
        print('Saving')


@app.post("/Generator_T2")
async def generate_t2(
    tumor_image_file: UploadFile = File(...),
):
    await preprocess_image_file(tumor_image_file)

    os.system('python test.py --gpu_ids -1 --name mri_t1t2_pix2pix --model pix2pix')

    with open('./results/clm.jpg', 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())

    return {
        'image': encoded_string.decode('utf-8')
    }

@app.post("/Generator_T1")
async def generate_t1(
    # tumor_image_file: UploadFile = File(...),
    tumor_image_path: str
):
    preprocess_image_file(tumor_image_path)

    os.system('python test.py --gpu_ids -1 --name mri_t2t1_pix2pix --model pix2pix')

    return {
        'file_path': './results/result.jpg'
    }

@app.post("/metric_images")
async def perform_metrics(
    # tumor_image_file: UploadFile = File(...),
    true_image_file: UploadFile = File(...),
):
    # generated_image = cv2.imread(generated_image_path, 0)
    true_image = await get_image_file(true_image_file)
    true_image = true_image[:, :, 0]
    true_image = cv2.resize(true_image, (256, 256))
    genereated_image = cv2.imread('./results/clm.jpg', 0)


    similarity = metric_ssim(true_image, genereated_image)


    return {
        'similarity': similarity
    }