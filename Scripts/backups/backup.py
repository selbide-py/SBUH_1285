from llama_cpp import Llama

print("Hey")

llm = Llama(model_path="C:\TheGoodShit\PythonProject\forMindEaseDocumentation\KoboldAI\models\Wizard-Vicuna-13B-Uncensored.ggmlv3.q4_1.bin",
                    # n_ctx = 4096,
                    # verbose= True
                    )

print("Model successfully loaded")
