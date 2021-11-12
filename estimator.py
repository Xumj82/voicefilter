import os
import glob
import torch
import random
import librosa
import argparse
import youtube_dl
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from noisereduce.generate_noise import band_limited_noise

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

def add_nosie(data,rate):
    noise_len = 2
    noise = band_limited_noise(min_freq=2000, max_freq= 12000, samples=len(data), samplerate=rate)*10
    noise_clip = noise[:rate*noise_len]
    audio_clip_band_limited = data+noise
    return audio_clip_band_limited

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)

def plot_wave(data,title='wave'):
    fig, ax = plt.subplots(figsize=(20,3))
    ax.set_title(title)
    ax.plot(data)

def mix(hp, audio, s1_dvec, s1_target, s2, vad_merge=False):
    srate = hp.audio.sample_rate

    d, _ = librosa.load(s1_dvec, sr=srate)
    w1, _ = librosa.load(s1_target, sr=srate)
    w2, _ = librosa.load(s2, sr=srate)
    assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
        'wav files must be mono, not stereo'

    # d, _ = librosa.effects.trim(d, top_db=20)
    # w1, _ = librosa.effects.trim(w1, top_db=20)
    # w2, _ = librosa.effects.trim(w2, top_db=20)

    # if reference for d-vector is too short, discard it
    if d.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
        return

    # LibriSpeech dataset have many silent interval, so let's vad-merge them
    # VoiceFilter paper didn't do that. To test SDR in same way, don't vad-merge.
    if vad_merge:
        w1, w2 = vad_merge(w1), vad_merge(w2)

    # I think random segment length will be better, but let's follow the paper first
    # fit audio to `hp.data.audio_len` seconds.
    # if merged audio is shorter than `L`, discard it
    L = int(srate * hp.data.audio_len)
    if w1.shape[0] < L or w2.shape[0] < L:
        return
    w1, w2 = w1[:L], w2[:L]

    mixed = w1 + w2

    mixed = add_nosie(mixed,srate)
    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, mixed = w1/norm, w2/norm, mixed/norm

    target_mag, _ = audio.wav2spec(w1)
    mixed_mag, _ = audio.wav2spec(mixed)

    return w1,w2,mixed

def download_audio_from_url(idx,start_time, end_time,url_to_video,path_to_video='data/'):
    dl_path = path_to_video+'video_dl_{0}.m4a'.format(idx)
    final_path = path_to_video+'video_{0}.m4a'.format(idx)
    if os.path.exists(dl_path):
        os.remove(dl_path)
    if os.path.exists(final_path):
        os.remove(final_path)
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'outtmpl': str(dl_path),
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_to_video])
        # ffmpeg_extract_subclip(dl_path , start_time, end_time, targetname=final_path)
        # if os.path.exists(dl_path):
        #     os.remove(dl_path)
    except Exception as e: # work on python 2.x
        print('video_{0} : {1}'.format(idx, str(e)))

def train_wrapper(train_folders):
    audio = Audio(hp)
    #从子文件中随机挑选音频作为d-vector, target 和 noise (每个子文件夹的音频来自同一人)
    train_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True) for spk in train_folders]
    spk1, spk2 = random.sample(train_spk, 2)
    s1_dvec, s1_target = random.sample(spk1, 2)
    s2 = random.choice(spk2)
    return mix(hp, audio, s1_dvec, s1_target, s2)

def audio_split(audio, segement_duration=30):
    srate = hp.audio.sample_rate
    segement_length = segement_duration*srate
    audio_length = len(audio)
    zeros  = np.zeros(segement_length-audio_length%segement_length)
    audio = np.concatenate((audio,zeros),axis=0)
    audio = audio.reshape((-1, segement_length))
    return audio


