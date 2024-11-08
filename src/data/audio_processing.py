import subprocess
import os
import yaml


config_path = "conf/dataset/ufc101.yaml"
with open(config_path, "r") as file:
    cfg = yaml.safe_load(file)


input_folder = cfg["input_folder"]
output_folder = cfg["output_folder_audio"]
bit_rate = cfg["bit_rate"]
sample_rate = cfg["sample_rate"]
channels = cfg["channels"]

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

