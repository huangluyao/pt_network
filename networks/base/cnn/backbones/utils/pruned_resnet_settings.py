import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

setting_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '_pruned_resnet_settings')
setting_files = os.listdir(setting_dir)

model_settings = dict()

for setting_file in setting_files:
    model_name = os.path.splitext(setting_file)[0]
    with open(os.path.join(setting_dir, setting_file), 'r') as f:
        model_settings[model_name] = json.load(f)