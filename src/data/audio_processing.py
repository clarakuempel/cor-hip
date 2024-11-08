import subprocess
import os
import yaml


config_path = "../../conf/dataset/ucf101.yaml"
with open(config_path, "r") as file:
    cfg = yaml.safe_load(file)

print(cfg)

input_folder = cfg["dataset"]["input_folder"]
output_folder = cfg["dataset"]["output_folder_audio"]
bit_rate = cfg["dataset"]["bit_rate"]
sample_rate = cfg["dataset"]["sample_rate"]
channels = cfg["dataset"]["channels"]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for file_name in os.listdir(input_folder):
    if file_name.endswith(".avi"):
        input_path = os.path.join(input_folder, file_name)
       	output_audio_path = os.path.join(output_folder, file_name.replace(".avi", ".wav"))

        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        command = f"ffmpeg -i {input_path} -ab {bit_rate} -ac {channels} -ar {sample_rate} -vn {output_audio_path}"
        subprocess.call(command, shell=True)


subprocess.call(command, shell=True)
