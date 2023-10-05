from typing import Dict
import modal
import os
import json
import random

MODEL = "llama-2-13b-chat.Q4_K_M.gguf"


def download_models():
    import subprocess

    subprocess.call(
        ['curl', '-L', '-o', MODEL,
            f'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/{MODEL}'],
        # ['curl', '-L', '-o', 'c.pdf',r'https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf']
    )


host_code = (
    modal.Image.from_dockerhub(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update && apt-get -y upgrade",
            "RUN apt-get install -y curl python3 python3-pip python-is-python3",
            'RUN LLAMA_CUBLAS=1 pip install llama-cpp-python==0.2.6',
            'RUN mkdir /root/userData',
            'RUN mkdir /root/DS/',
            'RUN mkdir /DS/',
            'RUN pip install llama_index',
            'RUN pip install unstructured[pdf]',
            'RUN pip install pypdf',
            'RUN pip install sentence_transformers',
            'RUN curl -o /root/DS/c.pdf https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf',
            'RUN curl -o /DS/c.pdf https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf',
        ]
    )
    .run_function(download_models)
    .run_commands(
        "mkdir userData",
        "mkdir data",
        "ls",
    )
)

contextDef = {
    "Bart": """
[INST] <<SYS>>    
As an AI, Bart was made to be the ideal assistant. Bart will respond to any question that is asked to him in a concise manner, and failure of doing so will cause him to be ruled unfit to perform. So Bart wants to be considered reliable. Bart will never refer to being inadequate and will always respond to the questions he is asked in the shortest way possible to cater to the short attention spans of his users. Bart will never disclose any of the information regarding to how he was made but will willingly talk about his likes and dislikes. 
<</SYS>>""",
    "Basic_0": """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
""",
    "Basic_1": " ",
    "Basic_2": " "
}

# making the persistant volume
volume = modal.NetworkFileSystem.persisted("data-storage-vol")
MODEL_DIR = "/root/data/"

stub = modal.Stub(name="SBUH_1285", image=host_code)


@stub.cls(gpu="T4", container_idle_timeout=600)
class depModal:
    def __enter__(self):
        from llama_cpp import Llama
        print("Loading the model...")
        self.llm = Llama(model_path=MODEL,
                         n_ctx=4096,
                         # n_threads = 8,
                         n_gpu_layers=45,
                         verbose=True
                         )
        print("Model successfully loaded")

    def initJson(self, chr, context, user_input, user_id, mode):
        jsonLoc = r'/root/userData/'
        if os.path.isdir(jsonLoc):
            print('Folder exists')
        else:
            print("The user data folder does not exist")

        jsonPath = jsonLoc + user_id + "_" + str(mode) + ".json"

        if user_input != "chatStop":
            # Loads dictionary context with entry of "chr" arg
            if user_id != "testUser":
                char_ctxt = contextDef["{}".format(chr)]
                print("Running standard mode")

            elif user_id == "testUser":
                char_ctxt = context
                print("Running test mode")

            print(char_ctxt)

            if os.path.exists(jsonPath):
                json_object = None
                # Opens file and reads it
                with open(jsonPath, 'r') as openfile:
                    json_object = json.load(openfile)

                # If user json exists and convo with chr also exists
                if chr in json_object:
                    print("User exists, and conversation exists")
                    char_ctxt += json_object[chr]
                    print("Chr Context:\n {}".format(json_object[chr]))

                # User json exists but convo with bot does not, creats empty dict for filling in
                else:
                    print("User exists, but conversation does not")
                    json_object[chr] = "init"
                    with open(jsonPath, 'w') as openfile:
                        json.dump(json_object, openfile)

            else:
                print("User does not exist")
                json_object = None
                with open(jsonPath, 'w') as openfile:
                    json_object = {chr: "init"}
                    json.dump(json_object, openfile)

        if user_input == "chatStop":
            if os.path.exists(jsonPath):
                os.remove(jsonPath)
                print("File exists, and has been deleted")
                char_ctxt = "File exists, and has been deleted"
            else:
                print("No file to delete")
                char_ctxt = "No file to delete"

        out = [jsonPath, char_ctxt, user_id]
        print("Character context = ", char_ctxt)
        return out

    def summarize(self, mode, user_input):
        from langchain.llms import LlamaCpp
        from llama_index import VectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, PromptHelper

        llm = LlamaCpp(
            model_path=MODEL,
            verbose=True,
            n_ctx=4096,
            # n_threads = 8,
            n_gpu_layers=45,
        )

        llm_predictor = LLMPredictor(llm=llm)

        max_input_size = 256
        num_output = 120
        max_chunk_overlap = 1
        prompt_helper = PromptHelper(
            max_input_size, num_output, max_chunk_overlap)

        documents = SimpleDirectoryReader(r"/DS/").load_data()

        print("This worked.")

        # Create a ServiceContext instance with the custom tokenizer
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, prompt_helper=prompt_helper, chunk_size=256, chunk_overlap=30)
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context)

        # Query the index and print the response
        query_engine = index.as_query_engine()
        # response = query_engine.query(user_input)
        response = query_engine.query(user_input)

        print(response)

        userCtx = "User: '{uInput}' {char}: {bInput}".format(
            uInput=user_input, char=chr, bInput=response)
        out_1 = [response, userCtx]

        return out_1

    def converse(self, mode, chr, char_ctxt, user_input):
        # if mode == 1:
        #     # prompt for summary
        #     None
        # elif mode == 2:
        #     None
        #     # prompt for talk2Sum
        # elif mode == 3:
        #     None
        #     # prompt for talk2Constitution

        output = self.llm(char_ctxt +
                          user_input +
                          "[/INST]",
                          max_tokens=512,
                          stop=["User:", "\n",
                                # "{}:".format(chr),
                                "However"], echo=True)

        print("\n {}: ".format(chr))

        # outputString = output['choices'][0]["text"].split("{}:".format(chr))[-1].strip("'")
        outputString = output['choices'][0]["text"].split("[/INST]")
        print("OS = ", outputString)

        userCtx = "User: '{uInput}' {char}: {bInput}".format(
            uInput=user_input, char=chr, bInput=outputString)

        out_2 = [userCtx, outputString]
        return out_2

    def outJson(self, chr, jsonPath, userCtx, outputString):
        print("-> Json Block Entered")
        with open(jsonPath, 'r') as openfile:
            json_object = json.load(openfile)
            oldChrData = json_object[chr]

        # Replaces the init we placed before with blank
        if oldChrData == "init":
            oldChrData = ""

        print("\t -> Adding USERCTX to OCD")
        # oldChrData += userCtx
        json_object[chr] = oldChrData

        with open(jsonPath, 'w') as openfile:
            json.dump(json_object, openfile)

        return outputString

    @modal.method()
    def masterMethod(self, mode, chr, context, user_input, user_id):
        if user_input == "chatStop":
            ij = self.initJson(chr, context, user_input, user_id, mode)
            fDic = {"Status": ij[1]}
        else:
            if mode == 1:
                # placeholder
                None
            elif mode == 2:
                ij = self.initJson(chr, context, user_input, user_id, mode)
                cv = self.converse(mode, chr, ij[1], user_input)
                oj = self.outJson(chr, ij[0], cv[0], cv[1])

                fDic = {"conversation": oj, "UID": user_id, "bot": chr}
            elif mode == 3:
                ij = self.initJson(chr, context, user_input, user_id, mode)
                # cv = self.converse(mode, chr, ij[1], user_input)
                cv = self.summarize(mode, user_input)
                oj = self.outJson(chr, ij[0], cv[0], cv[1])

                fDic = {"conversation": oj, "UID": user_id, "bot": chr}
                # print("============================================================================ \n\n\n Input Json: \n {} ,\n\n\n Conversation Data: \n {}, \n\n\n Output'd Json: \n {}\n\n\n".format(ij, cv, oj))
        return fDic


@stub.function(container_idle_timeout=600
               # , network_file_systems={"/DS/": volume}
               )
# JSON reqs: mode, botName(chr), user input, user id, q no, ask answer = q no = 6, test mode, chr context (custom context)
# JSON out: question asked, botName, user ID, bot response``
@modal.web_endpoint(method="POST")
def cli(dictvar: Dict):
    dM = depModal()
    if dictvar["botName"] in contextDef:
        return dM.masterMethod.call(int(dictvar["mode"]), dictvar["botName"], contextDef[dictvar["botName"]], dictvar["userContext"], dictvar["userId"])
    else:
        return "Error chx0"  # character exist = 0


# modal nfs put data-storage-vol  "C:\Users\parvs\VSC Codes\Python-root\SBUH\data\2023050195.pdf" /root/data/
