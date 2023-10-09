import modal
from typing import Dict

MODEL = "llama-2-13b-chat.Q4_K_M.gguf"


def download_models():
    import subprocess

    subprocess.call(
        ['curl', '-L', '-o', MODEL,
            f'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/{MODEL}']
    )


host_machine_code = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update && apt-get -y upgrade",
            "RUN apt-get install -y curl python3 python3-pip python-is-python3",
        ]
    )
    .run_function(download_models)
    .run_commands(
        "apt-get -y install cmake",
        "apt-get -y install protobuf-compiler",
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
                         n_gpu_layers=40,
                         n_ctx=4096,
                         verbose=True,
                         )

        print("Model successfully loaded\nAttempting generation")

    def textGen(self, context):
        output = self.llm(context,
                          max_tokens=256,
                          #   stop=["User:", "\n",
                          #         # "{}:".format(chr),
                          #         "However"],
                          echo=True
                          )

        xCont = output['choices'][0]['text']
        cCont = xCont.replace(context, "")
        contMain = [xCont, cCont]

        print(cCont)
        return contMain
    # TODO Need to make 2 different variables, xCont and cCont, and make them callable, clearly

    @modal.method()
    def runner(self):
        self.textGen(self, uN, au,)


@stub.function(gpu="T4", container_idle_timeout=600)
@modal.web_endpoint(method="POST")
def cli(varD: Dict):
    dM = depModal()

    # ! Input = userName, auth, qCont, mode
    # ! Output = userName, auth, cCont, xCont, mode

    # ! !!!! IMP, AUTH NEEDS TO BE DONE BY THE BACKEND, WILL NOT BE HANDLED BY THE AI BACKEND !!!!
    if varD['auth'] == 1:
        if varD['mode'] in [1, 2, 3, 4, 5]:
            return dM.runner.call(varD["Q"])
        else:
            return "AI Server Message > Error: This mode does not exist"
    else:
        return "AI Server Message > Error: Auth has failed"
