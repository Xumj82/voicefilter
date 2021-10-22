import os
import glob
import torch
import librosa
import argparse
import soundfile as sf

from utils.audio import Audio
from utils.hparams import HParam
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder

hp = HParam('config/default.yaml')

def init_model(checkpoint_path,embedder_path):
    model = VoiceFilter(hp).cuda()
    chkpt_model = torch.load(checkpoint_path)['model']
    model.load_state_dict(chkpt_model)
    model.eval()

    embedder = SpeechEmbedder(hp).cuda()
    chkpt_embed = torch.load(embedder_path)
    embedder.load_state_dict(chkpt_embed)
    embedder.eval()

    return model,embedder

def predict(reference_wav,
            mixed_wav,
            model,
            embedder
            ):
    with torch.no_grad():

        audio = Audio(hp)
        # dvec_wav, _ = librosa.load(reference_file, sr=16000)
        dvec_mel = audio.get_mel(reference_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float().cuda()
        dvec = embedder(dvec_mel)
        dvec = dvec.unsqueeze(0)

        # mixed_wav, _ = librosa.load(mixed_file, sr=16000)
        mag, phase = audio.wav2spec(mixed_wav)
        mag = torch.from_numpy(mag).float().cuda()

        mag = mag.unsqueeze(0)
        mask = model(mag, dvec)
        est_mag = mag * mask

        est_mag = est_mag[0].cpu().detach().numpy()
        est_wav = audio.spec2wav(est_mag, phase)

        return est_wav
        # os.makedirs(out_dir, exist_ok=True)
        # out_path = os.path.join(out_dir, 'result.wav')
        # # librosa.output.write_wav(out_path, est_wav, sr=16000)       
        # sf.write(out_path, est_wav, 16000)