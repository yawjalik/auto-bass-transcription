import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import math
import pickle
import librosa


class Vocab:
    def __init__(self):
        self.note2idx = {'{': 0, '}': 1, '<unk>': 2, '<pad>': 3}
        self.note2count = {'{': 0, '}': 0, '<unk>': 0, '<pad>': 0}
        self.idx2note = {0: '{', 1: '}', 2: '<unk>', 3: '<pad>'}
        self.n_notes = 4

    def add_notes(self, notes_str):
        for note in notes_str.split():
            self.add_note(note)

    def add_note(self, note):
        if note not in self.note2idx:
            self.note2idx[note] = self.n_notes
            self.note2count[note] = 1
            self.idx2note[self.n_notes] = note
            self.n_notes += 1
        else:
            self.note2count[note] += 1

    def notes_to_indices(self, notes: str):
        note_indices = []
        for note in notes.split(' '):
            if note in self.note2idx:
              note_indices.append(self.note2idx[note])
            else:
              note_indices.append(self.note2idx['<unk>'])
        return note_indices

    def indices_to_notes(self, indices):
        note_str = ''
        for i in indices:
            if i in self.idx2note:
                note_str += self.idx2note[i]
            else:
                note_str += '<unk>'
            note_str += ' '
        return note_str[:-1]


# https://github.com/openai/whisper/blob/main/whisper/model.py
# Used for audio positional embedding
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class PositionalEncoding(nn.Module):
    # Adds PE to the target label
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 100):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        """
        Arguments:
            token_embedding: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# Layer norm instead of batch norm
# From Whisper model
class AudioEncoder(nn.Module):
    def __init__(self, n_state):
        super().__init__()
        self.n_state = n_state
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_state, kernel_size=256, stride=4, padding=1).to(device)
        self.pool1 = nn.MaxPool1d(kernel_size=64, stride=64, padding=1)
        self.conv2 = nn.Conv1d(in_channels=n_state, out_channels=n_state, kernel_size=3, stride=1, padding=1).to(device)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.layer_norm = nn.LayerNorm(n_state).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        x = self.pool2(F.relu(x))
        x = self.layer_norm(x.permute(0, 2, 1))
        positional_embedding = sinusoids(x.shape[1], self.n_state).to(x.dtype).to(device)
        x = x + positional_embedding
        return x

class Transcriber(nn.Module):
    def __init__(self,
             n_state: int,
             vocab,
             dropout: float = 0.2):
        super().__init__()
        self.vocab = vocab
        self.audio_encoder = AudioEncoder(n_state)
        self.transformer = nn.Transformer(d_model=n_state,
                                          num_encoder_layers=6,
                                          num_decoder_layers=6,
                                          dropout=dropout,
                                          batch_first=True)
        # converts one-hot labels of (T, vocab_size) to (T, n_filters)
        self.target_embedding = nn.Embedding(vocab.n_notes, n_state)
        # adds positional encoding to the embedded target
        self.positional_encoding = PositionalEncoding(n_state, dropout)
        self.generator = nn.Linear(n_state, vocab.n_notes)

    def forward(self, src, target):
        src = self.audio_encoder(src)    # (1, S, n_filters)
        # src_mask = torch.zeros(src.shape[1], src.shape[1]).to(device)
        # target = (1, T - 1)
        tgt = self.target_embedding(target.to(device))  # (1, T - 1, n_filters)
        tgt = self.positional_encoding(tgt).to(device)  # (1, T - 1, n_filters)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        outs = self.transformer(src, tgt, tgt_mask=tgt_mask)
        outs = self.generator(outs)    # (1, seq, vocab_size)
        return outs
    

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Vocab':
            module = __name__
        return super().find_class(module, name)
    

device = torch.device('cpu')
model_dir = os.path.join(os.path.dirname(__file__), 'models')
samples_dir = os.path.join(os.path.dirname(__file__), 'samples')


def inference(model, audio: torch.Tensor) -> str:
    with torch.no_grad():
        model.eval()
        max_len = 20
        src = model.audio_encoder(audio.unsqueeze(0).to(device))  # (1, S, n_filters)
        memory = model.transformer.encoder(src)
        tgt = torch.zeros((1, max_len), dtype=torch.int32).to(device)
        tgt[0, 1:] = 2  # set first to SOS and the rest to <unk>
        tgt_mask = model.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        for i in range(max_len - 1):
            tgt_in = model.target_embedding(tgt)  # (1, max_len, n_filters)
            tgt_in = model.positional_encoding(tgt_in)  # (1, max_len, n_filters)
            out = model.transformer.decoder(tgt_in, memory, tgt_mask=tgt_mask)  # (1, max_len, n_filters)
            out = model.generator(out)  # (1, max_len, vocab)
            out = F.softmax(out.squeeze(0)[:i + 1], dim=1).argmax(dim=1)[-1]  # int
            tgt[0, i + 1] = out
            if out == 1:
                # Found end token, stop
                break

        print(tgt[0])
        predicted = model.vocab.indices_to_notes(tgt[0].tolist()).split('<unk>')[0].strip()
        return predicted


def to_full_score(notes_str, name='', version='2.24.3'):
    bpm = 120

    ly = f'''\\version "{version}"
    \\header {{
        title = "{name}"
        composer = "Auto Bass Transcriber"
    }}
    \\score {{
        \\layout {{
            \\tempo 4 = {bpm}
            \\clef bass
        }}

        \\new Staff {{
            \\absolute {notes_str}
        }}

        \\midi {{ }}
    }}
    '''
    return ly


def predict(audio: np.array, model_version: str = '3.0'):
    # Load model
    vocab = CustomUnpickler(open(os.path.join(model_dir, f'vocab-{model_version}.pkl'), 'rb')).load()
    model = Transcriber(n_state=256, vocab=vocab)
    model.load_state_dict(torch.load(os.path.join(model_dir, f'autobass-{model_version}.pth'), map_location=device))
    model.eval()

    audio = torch.tensor(audio).unsqueeze(0)
    print(audio.shape)
    predicted = inference(model, audio)
    # If no end token, append to prevent compilation error
    if predicted[-1] != '}':
        predicted += ' }'
    return to_full_score(predicted)


if __name__ == '__main__':
    # Load audio
    sample_wav = librosa.load(os.path.join(samples_dir, 'wav/0.wav'), sr=16000)[0]
    print(predict(sample_wav))
