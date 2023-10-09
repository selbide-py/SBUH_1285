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

    def textGen(self, context, mode):
        output = self.llm(context,
                          max_tokens=256,
                          #   stop=["User:", "\n",
                          #         # "{}:".format(chr),
                          #         "However"],
                          echo=True
                          )

        xCont = output['choices'][0]['text']
        cCont = xCont.replace(context, "")
        contMain = {"xCont": xCont, "cCont": cCont}

        print(cCont)
        return contMain

    def outputCleanup(self):
        # TODO Need to make a function that cleans up the output JSON before it gets sent, includes formatting and all that
        None

    @modal.method()
    def runner(self, uN, auth, qCont, mode):
        template = '''
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]
'''
        # ? use template.replace("{prompt}", <actual prompt here>)
        if mode == 1:
            # ! Basic const talk / dumb down
            # TODO Need to make it so that llamaIndex/langChain also get a different function to call, can call textGen inside that
            # TODO Need to add some form of limiters so that god mode(5) actually has some use
            self.textGen()
        elif mode == 5:
            x = self.textGen(template.replace("{prompt}", qCont))

        return x


@stub.function(gpu="T4", container_idle_timeout=600)
@modal.web_endpoint(method="POST")
def cli(varD: Dict):
    dM = depModal()

    # ! Input = userName, auth, qCont, mode
    # ! Output = userName, auth, cCont, xCont, mode
    # ? I think we can remove "userName", as the routing is the backend thing, the AI server has to do nothing with the name
    # ? of the user, all it cares about is that it has to generate, and that's it

    # ! !!!! IMP, AUTH NEEDS TO BE DONE BY THE BACKEND, WILL NOT BE HANDLED BY THE AI BACKEND !!!!
    if varD.len() == 5:
        if varD['auth'] == 1:
            if varD['mode'] in [1, 2, 3, 4, 5]:
                return dM.runner.call(varD["Q"])
            else:
                return "AI Server Message > Error: This mode does not exist"
        else:
            return "AI Server Message > Error: Auth has failed"
    else:
        return "AI Server Message > Error: Insufficient parameters"
