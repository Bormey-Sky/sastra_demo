import os
import math
import glob
import torch
import librosa
import numpy as np
import panns_inference
from sklearn.preprocessing import normalize
from panns_inference.inference import AudioTagging



def split_audio(audio, num_seconds = 1): 
    chunk_length = (32000 * num_seconds)
    num_chunks = math.ceil(audio.shape[0] / chunk_length)
    num_pad = (num_chunks * chunk_length) - audio.shape[0]
    audio = np.pad(audio, pad_width=(0, num_pad), mode="constant", constant_values=0)
    chunks = np.reshape(audio, (num_chunks, chunk_length))
    
    return chunks



if __name__ == "__main__":

    AUDIO_FOLDER = "songs/កូនស្រី.mp3"
    (audio, _) = librosa.core.load(AUDIO_FOLDER, sr=32000, mono=True)
    chunks = split_audio(audio, num_seconds=3)
    emb_model = AudioTagging(checkpoint_path=None, device='cuda')  
    # (clipwise_output, embedding) = emb_model.inference(audio)
    result = emb_model.inference(chunks)
    embedding = result[1]
    print(embedding.shape)

