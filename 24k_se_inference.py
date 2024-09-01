import soundfile
import torch
import librosa
import os
from model.aia_trans_onemic import DF_Serial_aia_complex_trans_ri
from model.stft import STFT

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path, dtype="float32")
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def ini_model(model_path):

    model = DF_Serial_aia_complex_trans_ri()

    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)
    model.to('cuda')
    model.eval()

    return model


def se_infer(wav_path, se_model):

    audio, fs = read_audio(wav_path)

    stft = STFT(
        filter_length=512,
        hop_length=256
    ).to('cuda')

    mixture = torch.from_numpy(audio).to('cuda')
    mixture = mixture.unsqueeze(0)

    mixture_D = stft.transform(mixture)

    spec_complex = torch.stack([mixture_D[:, :, :, 0], mixture_D[:, :, :, 1]], 1)

    est_real, est_imag = se_model(spec_complex)

    enhanced_D = torch.stack([est_real, est_imag], 3)
    enhanced = stft.inverse(enhanced_D)

    enhanced_audio = enhanced.detach().cpu().squeeze().numpy()

    return enhanced_audio, fs


if __name__ == '__main__':


    speech_dir = "D:\\JMCheng\\RT_24k\\DATASET\\testset\\noisy"

    workspace = "D:\\JMCheng\\RT_24k\\DATASET\\testset"

    pretrained_path = "model\\pretrained\\model_24k.pth"

    se_model = ini_model(pretrained_path)

    speech_names = []
    for dirpath, dirnames, filenames in os.walk(speech_dir):
        for filename in filenames:
             if filename.lower().endswith(".wav"):
                speech_names.append(os.path.join(dirpath, filename))

    for speech_na in speech_names:

        speech_na_basename = os.path.basename(speech_na)
        # Read and enhance speech

        enh_audio, fs = se_infer(speech_na, se_model)
        out_audio_path = os.path.join(workspace, "enhanced", "%s" % speech_na_basename)
        create_folder(os.path.dirname(out_audio_path))

        write_audio(out_audio_path, enh_audio, fs)