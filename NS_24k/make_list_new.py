import os
import librosa
import soundfile as sf
path1 = 'D:\\JMCheng\\RT_24k\\DATASET\\conf_to_cheng\\'

wav_names1 = os.listdir(path1)

index = 0

f = open('D:\\JMCheng\\RT_24k\\24k_SE_exp\\133h_24k_AntTTS_SE.lst', 'w')
for i in range(len(wav_names1)):
    speech_basename = os.path.basename(wav_names1[i])
    speech_fpart = os.path.splitext(speech_basename)[0]
    name_list = speech_fpart.split("_")
    if name_list[1] == "nearend":
        noisy, fs = sf.read(path1 + wav_names1[i])

        target_name = name_list[0] + "_" + "target" + ".wav"

        # clean, fs1 = sf.read(path1 + target_name)

        duration = len(noisy) / fs
        f.write(path1 + wav_names1[i] + ' ' + path1 + target_name + ' ' + '{}'.format(duration) + '\n')
        print("No.{} ".format(i) + wav_names1[i])
        index += 1
    if index == 80000:
        break
f.close()