import modal
# import llama_cpp

MODEL = "llama-2-13b-chat.Q4_K_M.gguf"


def download_models():
    import subprocess

    subprocess.call(
        ['curl', '-L', '-o', MODEL,
            f'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/{MODEL}']
    )


host_machine_code = (
    modal.Image.from_dockerhub(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update && apt-get -y upgrade",
            "RUN apt-get install -y curl python3 python3-pip python-is-python3",
        ]
    )
    .run_function(download_models)
    .run_commands(
        # "RUN sudo apt-get -y install cmake",
        "apt-get update && apt-get -y install cmake protobuf-compiler",
        'LLAMA_CUBLAS=1 CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.6',
    )
)

stub = modal.Stub(name="SBUH_1285", image=host_machine_code)


@stub.cls(gpu="T4", container_idle_timeout=600)
class depModal:
    def __enter__(self):
        from llama_cpp import Llama
        print("Loaded the library\nLoading the model....")
        self.llm = Llama(model_path=MODEL,
                         n_gpu_layers=10,
                         n_ctx=4096,
                         verbose=True,
                         )

        print("Model successfully loaded\nAttempting generation")

    def textGen(self):
        output = self.llm("What is onePlus ?",
                          max_tokens=512,
                          stop=["User:", "\n",
                                # "{}:".format(chr),
                                "However"], echo=True)

        print(output)
        return output

    @modal.method()
    def runner(self):
        self.textGen()


@stub.function(container_idle_timeout=600)
@modal.web_endpoint(method="POST")
def cli():
    dM = depModal()
    dM.runner()
