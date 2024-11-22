import os
import string

import numpy as np
import librosa
import sounddevice as sd
import whisper
import torch

from torch import nn
from queue import Queue

VOCABULARY = list(' ' + string.ascii_lowercase)
VOCAB_SIZE = len(VOCABULARY)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeedForwardModule(nn.Module):
    def __init__(self, input_dim : int, output_dim : int):
        super(FeedForwardModule, self).__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(input_dim, output_dim*4),
            nn.ReLU()
        )
        self.linear = nn.Linear(output_dim*4, output_dim)

    def forward(self, x):
        x = self.linear_relu(x)
        x = self.linear(x)
        return x

class ConformerModule(nn.Module):
    def __init__(self, embedding_dim : int, output_dim : int, num_heads : int, conv_kernel_size : int = 5, dropout_rate : float = 0.1):
        super(ConformerModule, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads)
        self.feed_forward1 = FeedForwardModule(embedding_dim, embedding_dim)
        self.conv_block = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim*2, 1),
            nn.ReLU(),
            nn.Conv1d(embedding_dim*2, embedding_dim, conv_kernel_size, padding=(conv_kernel_size-1)//2),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward2 = FeedForwardModule(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attention_output)

        ff_output1 = self.feed_forward1(x)
        x = x + self.dropout(ff_output1)

        x = x.permute(1, 2, 0)
        conv_output = self.conv_block(x)
        x = x + self.dropout(conv_output)
        x = x.permute(2, 0, 1)
        x = self.layer_norm(x)

        ff_output2 = self.feed_forward2(x)
        x = x + self.dropout(ff_output2)
        return x

class ASRModel(nn.Module):
    def __init__(self, n_mels : int, hidden_dim : int, vocab_size : int, dropout_rate : float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, hidden_dim, 1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.conformer_block = nn.Sequential(
            ConformerModule(hidden_dim, hidden_dim, 8, dropout_rate=dropout_rate),
            ConformerModule(hidden_dim, hidden_dim, 8, dropout_rate=dropout_rate),
            ConformerModule(hidden_dim, hidden_dim, 8, dropout_rate=dropout_rate),
            ConformerModule(hidden_dim, hidden_dim, 8, dropout_rate=dropout_rate),
            ConformerModule(hidden_dim, hidden_dim, 8, dropout_rate=dropout_rate)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x = self.layer_norm1(x)
        x = self.conformer_block(x)
        x = self.layer_norm2(x)
        x = self.linear(x)
        return x

def load_model(model_path):
    # model has about 37 million parameters
    model = ASRModel(80, 512, VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

SAMPLE_RATE = 16000
CHUNK_DURATION = 2
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
CHANNELS = 1  # Моно

audio_queue = Queue()

def audio_to_mel(audio_array : np.ndarray, sample_rate : int, n_mels : int = 80, target_sample_rate : int = 16000):
        if sample_rate != target_sample_rate:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sample_rate)
        mel_spectr = librosa.feature.melspectrogram(y=audio_array, sr=target_sample_rate, n_mels=n_mels)
        log_mel_spectr = librosa.power_to_db(mel_spectr, ref=np.max)
        log_mel_spectr_norm = librosa.util.normalize(log_mel_spectr, axis=0)
        return log_mel_spectr_norm

def process_audio_chunk(chunk, model):
    chunk_mel = audio_to_mel(chunk, SAMPLE_RATE)
    with torch.no_grad():
        pred = model(torch.Tensor([chunk_mel]).to(device))
        pred = nn.Softmax(dim=2)(pred)
        chosen_tokens = torch.argmax(pred, dim=2)
    return chosen_tokens

def process_audio_chunk_whisper(chunk, model):
    audio_tensor = np.array(chunk, dtype=np.float32)
    result = model.transcribe(audio_tensor)
    return result["text"]

def decode_text(text_tokens):
    decoded_text = "".join(list(map(lambda x : VOCABULARY[int(x)], text_tokens)))
    text_formatted = ""
    for i, sym in enumerate(decoded_text):
        if i == 0 or (decoded_text[i] != decoded_text[i-1]):
            text_formatted += sym
    return text_formatted

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Статус ошибки: {status}")
    audio_queue.put(indata[:, 0])

def play_audio(chunk, sample_rate=SAMPLE_RATE):
    sd.play(chunk, samplerate=sample_rate)
    sd.wait()

def main():
    model_path = os.path.join('./checkpoint', 'asr_conformer_best.pth')
    model = load_model(model_path)
    prev = np.zeros((CHUNK_SIZE//8,))
    print("Transcription is started!")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=CHUNK_SIZE):
        while True:
            if not audio_queue.empty():
                audio_chunk = audio_queue.get()
                concated = np.concatenate([prev, audio_chunk], axis=0)
                transcription = decode_text(process_audio_chunk(concated, model))
                print(transcription, end=' ', flush=True)
                prev = audio_chunk[-CHUNK_SIZE//8:]

def main_whisper():
    # tiny.en has 39 million parameters
    model = whisper.load_model('tiny.en').to(device)
    prev = np.zeros((CHUNK_SIZE//8,))
    print("Transcription is started!")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=CHUNK_SIZE):
        while True:
            if not audio_queue.empty():
                audio_chunk = audio_queue.get()
                concated = np.concatenate([prev, audio_chunk], axis=0)
                transcription = process_audio_chunk_whisper(concated, model)
                print(transcription, end=' ', flush=True)
                prev = audio_chunk[-CHUNK_SIZE//8:]

if __name__ == "__main__":
    main_whisper()
