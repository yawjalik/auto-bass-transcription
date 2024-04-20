from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import base64
import librosa
import subprocess
from app.transcriber import predict, to_full_score

app = FastAPI(
    title='Automatic Bass Transcriber API',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

base_dir = os.path.dirname(__file__)
sample_audio_dir = os.path.join(base_dir, 'samples/wav')
sample_label_dir = os.path.join(base_dir, 'samples/labels')


class TranscribeRequest(BaseModel):
    name: str
    audio: str
    model: str


@app.get('/_healthcheck')
def healthcheck():
    return {'status': 'ok'}


@app.get('/samples')
def get_samples():
    sample_audio = os.listdir(sample_audio_dir)
    sample_label = os.listdir(sample_label_dir)

    samples = []
    for audio, label in zip(sample_audio, sample_label):
        sample = {'name': audio}
        # Load audio file and base64 encode
        audio_f = open(os.path.join(sample_audio_dir, audio), 'rb')
        audio_b64 = base64.b64encode(audio_f.read()).decode('utf-8')
        sample['audio'] = audio_b64
        audio_f.close()
        # Load label file
        label_f = open(os.path.join(sample_label_dir, label), 'r')
        label = label_f.read()
        sample['lilypond'] = to_full_score(label, name=audio.split('.')[0])
        label_f.close()
        samples.append(sample)

    return samples


@app.get('/models')
def get_models():
    models = list(filter(lambda x: x.endswith('.pth'), os.listdir(os.path.join(base_dir, 'models'))))
    # Extract version: name-version.pth
    models = list(map(lambda x: x.split('-')[1][: -4], models))
    print(models)
    return models


@app.post('/transcribe')
def transcribe(req: TranscribeRequest):
    name = req.name.split('.')[0]
    model_version = req.model

    # Write to tmp
    if not os.path.exists(os.path.join(base_dir, 'tmp')):
        os.makedirs(os.path.join(base_dir, 'tmp'))
    
    with open(os.path.join(base_dir, 'tmp', req.name), 'wb') as audio_f:
        audio_f.write(base64.b64decode(req.audio))
    # Load
    audio = librosa.load(os.path.join(base_dir, 'tmp', req.name), sr=16000)[0]

    score = predict(audio, model_version)
    # Save to file
    with open(os.path.join(base_dir, 'tmp', name + '.ly'), 'w') as f:
        f.write(score)

    # Generate pdf
    subprocess.run(['lilypond', '-o', os.path.join(base_dir, 'tmp', name), os.path.join(base_dir, 'tmp', name + '.ly')])

    pdf_f = open(os.path.join(base_dir, 'tmp', name + '.pdf'), 'rb')
    pdf_b64 = base64.b64encode(pdf_f.read()).decode('utf-8')
    f.close()

    return {'name': req.name, 'lilypond': score, 'pdf': pdf_b64}
