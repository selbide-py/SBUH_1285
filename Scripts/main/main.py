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
            'RUN mkdir /root/DS/',
            'RUN mkdir /DS/',
            'RUN curl -o /root/DS/c.pdf https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf',
            'RUN curl -o /DS/c.pdf https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf',
        ]
    )
    .run_function(download_models)
    .run_commands(
        "apt-get -y install cmake",
        "apt-get -y install protobuf-compiler",
        "apt-get update && apt-get -y install cmake protobuf-compiler",
        'LLAMA_CUBLAS=1 CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.6',
        'pip install langchain'
    )
)

# Flow
# __enter__(always runs on first run) --> {textGen --> outputCleanup}[All run via runner]

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

    def textGen(self, cCont, qCont, temp=0.6):
        context = cCont + qCont
        output = self.llm(context,
                          max_tokens=256,
                          #   stop=["User:", "\n",
                          #         # "{}:".format(chr),
                          #         "However"],
                          temperature=temp,
                          echo=True,
                          )

        xCont = output['choices'][0]['text']
        cCont = xCont.replace(context, "")
        contMain = {"xCont": xCont, "cCont": cCont}

        # print(cCont)
        return contMain

    def lcGen(mode, qCont):
        from langchain.llms import LlamaCpp
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

        template = """Question: {question}
        Answer: Sure - """
        prompt = PromptTemplate(
            template=template, input_variables=["question"])
        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Make sure the model path is correct for your system!
        llm = LlamaCpp(
            model_path=MODEL,
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            callback_manager=callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
        )

        llm_chain = LLMChain(prompt=prompt, llm=llm)
        # TODO Import the llm object from the textGen function
        # TODO check how to actually work out LLAMACPP via LC and if it's any different from native LCPP
        # TODO check how does the "QUESTION" input_variable work, and how does it work
        # TODO what does callback_manager do ?
        question = qCont
        llm_chain.run(question)

    def outputCleanup(self, userName, contMain, mode):
        # TODO Need to make it go through formatting of the output
        x = {"userName": userName,
             "xCont": contMain["xCont"],
             "cCont": contMain["cCont"],
             "mode": mode}
        return x

    @modal.method()
    def runner(self, userName, cCont, qCont, mode):
        template = '''
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]
'''
        # ? use template.replace("{prompt}", <actual prompt here>)
        if mode == 1:
            # ! Basic const talk / dumb down
            # TODO Need to make it so that llamaIndex/langChain also get a different function to call, can call textGen inside that
            # TODO Need to add some form of limiters so that god mode(5) actually has some use
            self.textGen()
        elif mode == 2:
            # ! Summarisation
            None
        elif mode == 5:
            prompt = template.replace("{prompt}", qCont)
            y = self.textGen(cCont, prompt)
            print(y)
            x = self.outputCleanup(userName, y, mode)

        return x


@stub.function(gpu="T4", container_idle_timeout=600)
@modal.web_endpoint(method="POST")
def cli(varD: Dict):
    dM = depModal()

    # ! Input = userName, auth, cCont, qCont, mode
    # ! Output = userName, cCont, xCont, mode
    # ? I think we can remove "userName", as the routing is the backend thing, the AI server has to do nothing with the name
    # ? of the user, all it cares about is that it has to generate, and that's it

    # ! !!!! IMP, AUTH NEEDS TO BE DONE BY THE BACKEND, WILL NOT BE HANDLED BY THE AI BACKEND !!!!
    try:
        if len(varD) == 5:
            if int(varD['auth']) == 1:
                if int(varD['mode']) in [1, 2, 3, 4, 5]:
                    return dM.runner.call(varD["userName"], varD["cCont"], varD["qCont"], int(varD["mode"]))
                else:
                    return "AI Server Message > Error: This mode does not exist"
            else:
                return "AI Server Message > Error: Auth has failed"
        else:
            return "AI Server Message > Error: Insufficient parameters"
    except:
        return "AI Server Message > Error: Code ran into some issues"
