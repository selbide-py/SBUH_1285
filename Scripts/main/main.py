from typing import Dict
import modal

MODEL = "llama-2-13b-chat.Q4_K_M.gguf"


def download_models():
    import subprocess

    subprocess.call(
        ['curl', '-L', '-o', MODEL,
            f'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/{MODEL}']
    )

# host_machine_code = (
#     modal.Image.from_registry(

    # "nvidia/cuda:12.1.1-devel-ubuntu22.04",
    # setup_dockerfile_commands=[


host_machine_code = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
    )
    .run_commands(
        "apt-get update && apt-get -y upgrade",
        "apt-get install -y curl python3 python3-pip python-is-python3",
        'pip install gdown',
        'mkdir /root/DS/',
        'mkdir /DS/',
        'curl -o /root/DS/c.pdf https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf',
        'curl -o /DS/c.pdf https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf',
        'mkdir /root/DS/legalLm/',
        'cd /root/DS/legalLm/',
        'ls',
        'gdown https://drive.google.com/uc?id=1mRsrXc9d2Tb7Wr2RXJJNDHtxMZa1osnY',
        'gdown https://drive.google.com/uc?id=1g6n6bEua8Lobs-2v2hSph0c8hDQu6Evg',
        'gdown https://drive.google.com/uc?id=1cyNsDAy90UMg0TIL3Sgv87VP4FQeJSZN',
        'gdown https://drive.google.com/uc?id=1LNO25lfqeAUsMasD0MwBCPO6w4_GJ-Uo',
        'gdown https://drive.google.com/uc?id=12IlRpbFnRpKvfh7-4wpicV7q4RUrUhSw',
        'gdown https://drive.google.com/uc?id=1-GGNsmtu7RIGDmkid4-CzP3ZVuu6BxHo',
        'gdown https://drive.google.com/uc?id=1-C0wB9h1nNfGTzIVx5P9A-USyZcqpDoB',
        'gdown https://drive.google.com/uc?id=1-71kufADT-LW0z7dgnuh6NFWdH9bIHW7',
        'ls',
    )
    .run_function(
        download_models
    )
    .pip_install(
        "torch", "transformers", "sentencepiece", "langchain"
    )
    .run_commands(
        "apt-get -y install cmake",
        "apt-get -y install protobuf-compiler",
        "apt-get update && apt-get -y install cmake protobuf-compiler",
        'LLAMA_CUBLAS=1 CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.6',
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


@stub.cls(gpu="T4", container_idle_timeout=600)
class sumModal:
    def __enter__(self):
        import os
        import torch
        import torch.quantization
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = torch.device("cuda")
        current_device = torch.cuda.current_device()

        print("Currently selected device:", current_device)
        print("CUDA is available: ", torch.cuda.is_available())
        print("Number of CUDA devices: ", torch.cuda.device_count())
        print("CUDA current device: ", torch.cuda.current_device())
        print("CUDA device name: ", torch.cuda.get_device_name(0))

        print("CUDA loaded\n")

        # Changing directory to root
        import subprocess
        wd = os.getcwd()
        os.chdir("/")
        subprocess.Popen("ls")
        os.chdir(wd)

        # Load Legal LLAMA tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/', local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            '/', local_files_only=True)
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8)

        # Move model to CUDA device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        print("Model loaded\n")

    def outputCleanup(self, userName, contMain, mode):
        # TODO Need to make it go through formatting of the output
        x = {"userName": userName,
             "xCont": contMain,
             "cCont": "null",
             "mode": mode}
        return x

    @modal.method()
    def runner(self, userName, cCont, qCont, mode):
        import torch

        print("Summarisation runner called")

        test_summaries = """When item 40 of Part 13 of the Schedule to the Constitution (Scheduled Castes) Order, 1950, declared "Sunris excluding Sahas" as a Schduled Caste, it indicates that men of Sunri caste but not those within that caste who formed the smaller caste group of Sahas, are members of a Scheduled Caste.
It does not indicate that Sahas are a caste distinct from the Sunri caste, nor was it intended to exclude from Sunris those members of that caste who bore the surname Saha.
[391 A, D].
Therefore, when the respondent challenged the election to the West Bengal Legislative Assembly, of the appellant who described himself as a member of the Sunri caste, on the ground that he was a member of the Saha caste group but failed to prove the allegation, it must be held that the appellant was a Sunri by caste and belonged to the Scheduled caste specified in the item, even though he bore the surname Saha.
[392 D].
"""
        # Assuming your test data is similarly prepared as train_tokenized
        test_tokenized = self.tokenizer(
            cCont,
            text_target=test_summaries,  # Use 'text_target' for the target sequences
            padding=True,
            truncation=True,
            max_length=512+256,  # Adjust as needed
            return_tensors='pt'
        )

        # Move test data to CUDA device
        test_tokenized = {key: val.to(self.device)
                          for key, val in test_tokenized.items()}

        with torch.no_grad():
            outputs = self.model.generate(**test_tokenized)
            generated_summaries = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True)

        # Print the generated summaries
        for summary in generated_summaries:
            # print(summary)
            None

        final_output_list, final_output = summary.split(". "), ""
        for i in final_output_list[:-1]:
            final_output += i + ". "
        print(final_output)

        return self.outputCleanup(userName, final_output, mode)


@stub.function(gpu="T4", container_idle_timeout=600)
@modal.web_endpoint(method="POST")
def cli(varD: Dict):
    # ! Seeing that if not invoking this class will not call the funcion class
    # ! Input = userName, auth, cCont, qCont, mode
    # ! Output = userName, cCont, xCont, mode
    # ? I think we can remove "userName", as the routing is the backend thing, the AI server has to do nothing with the name
    # ? of the user, all it cares about is that it has to generate, and that's it

    # !!!! IMP, AUTH NEEDS TO BE DONE BY THE BACKEND, WILL NOT BE HANDLED BY THE AI BACKEND !!!!
    try:
        if len(varD) == 5:
            if int(varD['auth']) == 1:
                if int(varD['mode']) in [1, 2, 3, 4, 5]:
                    dM = depModal()
                    return dM.runner.remote(varD["userName"], varD["cCont"], varD["qCont"], int(varD["mode"]))
                else:
                    return "AI Server Message > Error: This mode does not exist"
            # !! auth = 2 is for the AI summary server
            elif int(varD['auth']) == 2:
                sM = sumModal()
                return sM.runner.remote(varD["userName"], varD["cCont"], varD["qCont"], int(varD["mode"]))
            else:
                return "AI Server Message > Error: Auth has failed"
        else:
            return "AI Server Message > Error: Insufficient parameters"
    except Exception as e:
        return "AI Server Message > Error: Code ran into some issues {}".format(e)
