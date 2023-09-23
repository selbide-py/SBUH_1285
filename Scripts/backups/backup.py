from llama_cpp import Llama

print("Hey")

llm = Llama(model_path= "C:\TheGoodShit\PythonProject\SBUH\llama-2-13b-chat.Q4_K_M.gguf",
            n_gpu_layers=10
            # n_ctx = 4096,
            # verbose= True
            )

print("Model successfully loaded")