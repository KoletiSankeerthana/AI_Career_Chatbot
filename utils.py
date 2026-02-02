import json
import os
from dotenv import set_key

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return default
    return default

def save_api_key(key, env_path=".env"):
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write(f"GROQ_API_KEY={key}\n")
    else:
        set_key(env_path, "GROQ_API_KEY", key)
