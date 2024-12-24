import os
import glob
import math
import shutil
import librosa
import numpy as np
import soundfile as sf
from typing import Union
from pymilvus import Collection
from init_db import DatabaseClient
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, Body
from panns_inference.inference import AudioTagging
from fastapi.responses import HTMLResponse, FileResponse


ml_model = {}
global_values = {}


def split_audio(audio, num_seconds=1):
    chunk_length = (32000 * num_seconds)
    num_chunks = math.ceil(audio.shape[0] / chunk_length)
    num_pad = (num_chunks * chunk_length) - audio.shape[0]
    audio = np.pad(audio, pad_width=(0, num_pad),
                   mode="constant", constant_values=0)
    chunks = np.reshape(audio, (num_chunks, chunk_length))

    return chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("server starting")
    ml_model["audio_embeddor"] = AudioTagging(
        checkpoint_path=None, device='cuda')
    db_client = DatabaseClient(
        host="134.209.97.57",
        port="19530",
    )
    db_client.connect()
    db_client.load_collections()
    global_values["db_client"] = db_client
    yield
    db_client.release_collections()
    db_client.disconnect()
    ml_model.clear()
    global_values.clear()
    print("server closing")

app = FastAPI(
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/out", StaticFiles(directory="out"), name="out")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content


@app.get("/play/{filename}")
def play_audio(filename: str):
    filepath = os.path.join("out", filename)
    return FileResponse(filepath, media_type="audio/*")


@app.post("/api/register")
async def register(song: UploadFile, song_title: str = Body()):
    print("song title >> ", song.filename)
    emb_model = ml_model["audio_embeddor"]
    db_client: DatabaseClient = global_values["db_client"]
    content = await song.read()
    with open("out.mp3", "wb") as outfile:
        outfile.write(content)

    (audio, _) = librosa.core.load("out.mp3", sr=32000, mono=True)
    _, embedding = emb_model.inference(audio[None, :])
    song_id = db_client.insert(
        "songs",
        [
            {
                "embedding": embedding[0],
                "song_title": song.filename,
                "start_position_sec": 0,
                "end_position_sec": int(math.ceil(audio.shape[0] / 32000)),
                "song_id": 0,
                "is_chunk": False,
                "audio_url": song.filename,

            }
        ]
    )
    song_id = song_id["ids"][0]
    print(song_id)
    chunks = split_audio(audio, num_seconds=3)
    _, embeddings = emb_model.inference(chunks)
    print(embeddings.shape)
    db_client.insert(
        "songs",
        [
            {
                "embedding": embedding,
                "song_title": song_title,
                "start_position_sec": i * 3,
                "end_position_sec": (i + 1) * 3,
                "song_id": song_id,
                "is_chunk": True,
                "audio_url": song.filename,
            }
            for i, embedding in enumerate(embeddings)
        ]
    )


@app.post("/api/search")
async def search(song: UploadFile):

    # remove files from out folder
    for file_path in glob.glob("out/*"):
        os.remove(file_path)

    emb_model = ml_model["audio_embeddor"]
    db_client: DatabaseClient = global_values["db_client"]
    print(song)
    content = await song.read()
    with open("out.mp3", "wb") as outfile:
        outfile.write(content)

    (audio, _) = librosa.core.load("out.mp3", sr=32000, mono=True)
    _, embedding = emb_model.inference(audio[None, :])
    collection = Collection("songs")
    result = collection.search(
        data=[embedding[0]],
        anns_field="embedding",
        limit=10,
        param={
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        },
        expr="is_chunk == False",
        output_fields=['song_id', 'song_title',
                       'start_position_sec', 'end_position_sec', 'audio_url']

    )
    result = result[0]
    print(result)
    if len(result) == 0:
        return []

    if result[0].distance >= 0.9999999:

        db_filename = result[0].get("audio_url")
        original_path = f"songs/{db_filename}"

        out_path = f"out/{db_filename}"
        shutil.copyfile(original_path, out_path)

        web_url = f"/out/{db_filename}"
        return [{
            "song_id": result[0].id,
            "song_title": result[0].get("song_title"),
            "matching_score": result[0].distance,
            "match_segments": [{
                "matching_score": result[0].distance,
                "starting_position": result[0].get("start_position_sec"),
                "ending_position": result[0].get("end_position_sec"),
                "target_audio_url": web_url
            }]
        }]

    search_chunks = split_audio(audio, num_seconds=3)
    _, embeddings = emb_model.inference(search_chunks)
    results = collection.search(
        data=embeddings,
        anns_field="embedding",
        limit=10,
        param={
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        },
        expr="is_chunk == True",
        output_fields=['song_id', 'song_title',
                       'start_position_sec', 'end_position_sec', 'audio_url']

    )

    # threshold = 0.90
    out = {}

    for i, result in enumerate(results):
        if len(result) == 0:
            continue

        result = sorted(result, key=lambda x: x.distance, reverse=True)

        song_id = result[0].get("song_id")
        song_title = result[0].get("song_title")
        start_position_sec = result[0].get("start_position_sec")
        end_position_sec = result[0].get("end_position_sec")
        query_audio_chunk = search_chunks[i]
        target_audio, _ = librosa.core.load(
            f"songs/{result[0].get('audio_url')}", sr=32000, mono=True)
        target_audio_chunk = target_audio[start_position_sec *
                                          32000: end_position_sec * 32000]
        target_audio_url = f'out/{song_title}_{start_position_sec}_{end_position_sec}.wav'
        query_audio_url = f'out/{song_title}_{start_position_sec}_{end_position_sec}_match_{i}.wav'
        sf.write(target_audio_url, data=target_audio_chunk,
                 samplerate=32000, subtype="PCM_24")
        sf.write(query_audio_url, data=query_audio_chunk,
                 samplerate=32000, subtype="PCM_24")

        web_target_url = f"/{target_audio_url}"
        web_query_url = f"/{query_audio_url}"

        if song_id in out:
            out[song_id]["match_segments"].append({
                "query_audio_segment": i,
                "matching_score": result[0].distance,
                "starting_position": result[0].get("start_position_sec"),
                "ending_position": result[0].get("end_position_sec"),
                "target_audio_url": web_target_url,
                "query_audio_url": web_query_url
            })
            continue
        out[song_id] = {
            "song_id": song_id,
            "song_title": result[0].get("song_title"),
            "matching_score": result[0].distance,
            "match_segments": [{
                "query_audio_segment": i,
                "matching_score": result[0].distance,
                "starting_position": result[0].get("start_position_sec"),
                "ending_position": result[0].get("end_position_sec"),
                "target_audio_url": web_target_url,
                "query_audio_url": web_query_url
            }]
        }

    return out
