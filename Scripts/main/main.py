import modal

MODEL = "llama-2-13b-chat.Q4_K_M.gguf"

def download_models():
    import subprocess

    subprocess.call(
        ['curl', '-L', '-o', MODEL, f'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/{MODEL}']
        )
    
host_machine_code = 