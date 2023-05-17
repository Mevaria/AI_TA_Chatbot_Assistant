from langchain import LLMChain
from langchain.memory import ConversationSummaryMemory, ConversationBufferWindowMemory
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor
from llama_index.langchain_helpers.agents.toolkits import LlamaToolkit
import sys
import os
from langchain.llms import OpenAI
from llama_index import StorageContext, load_index_from_storage
from llama_index.langchain_helpers.agents import create_llama_chat_agent

os.environ["OPENAI_API_KEY"] = 'ADD API KEY HERE'
root_dir = os.getcwd()


def flatten(list):
    return [item for sublist in list for item in sublist]


def get_files(directory_path):
    total_files = []
    os.chdir(root_dir + "\\" + directory_path)
    docs_dir = os.getcwd()
    for root, dr, f in os.walk(docs_dir):
        for d in dr:
            total_files.append(SimpleDirectoryReader(d, recursive=True).load_data())
    return flatten(total_files)


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 1024
    max_chunk_overlap = 20
    chunk_size_limit = 1024

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = get_files(directory_path)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    os.chdir(root_dir)
    index.storage_context.persist()

    return index


def upload_document(doc):
    new_doc = SimpleDirectoryReader(doc).load_data()
    index.insert(new_doc[0])


def chatbot(input_text):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)

    tools = [
        Tool(
            name="GPT Index",
            func=lambda q: str(index.as_query_engine().query(q)),
            description="Useful when you need to ask questions about certain topics.",
            return_direct=True
        ),
    ]
    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
    llm_chain = LLMChain(llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo"), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
    response = agent_chain.run(input=input_text)
    return f"Agent: {response}"

def chatbot2(input_text):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
    tools = [
        Tool(
            name="GPT Index",
            func=lambda q: str(index.as_query_engine().query(q)),
            description="Useful when you need to ask questions about certain topics.",
            return_direct=True
        ),
    ]
    tool = LlamaToolkit(index_cofig=tools)
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
    agent_chain = create_llama_chat_agent(
        tool,
        llm,
        memory=memory,
    )
    return agent_chain.run(input=input_text)


index = construct_index("docs")

