import os
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

def instantiate_llm():
    print("Instantiating LLM...")
    model_path = "./models/Meta-Llama-3-8B-Instruct-Q6_K.gguf"

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=64,
        temperature=0.2,
        n_threads=os.cpu_count(),
        max_tokens=128,
        n_ctx=1024,
        verbose=False,
    )

    return llm

def build_pw_prompt():
    rl_template = """<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant capable of reading and understanding item specifications.
    I will provide you with two item specifications, each containing several attributes: your task is to determine if they refer to the same real-world entity.

    If you think that the specifications refer to the same entity, respond with "MATCH"; otherwise, even if you are unsure, respond with "REJECT". 
    Note: If an attribute value is 'nan', consider it as 'not present'.
    Don't add any additional information to your response.

    I'm giving you an example of a 'MATCH' pair and a 'REJECT' pair to help you understand the task better.
    ITEM 1:
    **Full Name**: Mario Rossi
    **Address**: Via Garibaldi 10, Roma
    **Birth Date**: 1980-05-12
    **Phone Number**: 3456789101

    ITEM 2:
    **Full Name**: M. Rossi
    **Address**: V. Garibaldi 10, 00184, Roma
    **Birth Date**: nan
    **Phone Number**: +39-345-678-9101

    EXPECTED RESPONSE: 'MATCH'

    ITEM 1:
    **Full Name**: Luigi Bianchi
    **Address**: Via Verdi 15
    **Birth Date**: nan
    **Phone Number**: +39-366-678-9101
    
    ITEM 2:
    **Full Name**: Luigi Bianchi
    **Address**: Via Garibaldi 10
    **Birth Date**: nan
    **Phone Number**: +39-366-678-9100 

    EXPECTED RESPONSE: 'REJECT'
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Question: {question}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    Answer: """

    rl_prompt = PromptTemplate.from_template(rl_template)
    return rl_prompt

def query_llm(llm, prompt, DATASET, instances, i1, i2):
    match DATASET:
        case "Abt_Buy":
            response = query_llm_abt_buy(llm, prompt, instances, i1, i2)
        case "Beers":
            response = query_llm_beers(llm, prompt, instances, i1, i2)
        case "DBLP_ACM":
            response = query_llm_dblp_acm(llm, prompt, instances, i1, i2)
        case _:
            raise ValueError("Invalid Dataset: choose among the Datasets in './datasets/'")
    
    return response

def query_llm_abt_buy(llm, prompt, instances, i1, i2):
    request = f"""Are the following item specifications related to the same entity?
    ITEM 1:
    **Name**: {instances.at[i1, 'name']}
    **Description**: {instances.at[i1, 'description']}
    **Price**: {instances.at[i1, 'price']}

    ITEM 2:
    **Name**: {instances.at[i2, 'name']}
    **Description**: {instances.at[i2, 'description']}
    **Price**: {instances.at[i2, 'price']}
    """

    chain = ({"question": RunnablePassthrough()} | prompt | llm)
    response = chain.invoke(request)
    return response

def query_llm_beers(llm, prompt, instances, i1, i2):
    request = f"""Are the following item specifications related to the same entity?
    ITEM 1:
    **Beer Name**: {instances.at[i1, 'Beer_Name']}
    **Brew_Factory_Name**: {instances.at[i1, 'Brew_Factory_Name']}
    **Style**: {instances.at[i1, 'Style']}
    **ABV**: {instances.at[i1, 'ABV']}

    ITEM 2:
    **Beer Name**: {instances.at[i2, 'Beer_Name']}
    **Brew_Factory_Name**: {instances.at[i2, 'Brew_Factory_Name']}
    **Style**: {instances.at[i2, 'Style']}
    **ABV**: {instances.at[i2, 'ABV']}
    """

    chain = ({"question": RunnablePassthrough()} | prompt | llm)
    response = chain.invoke(request)
    return response

def query_llm_dblp_acm(llm, prompt, instances, i1, i2):
    request = f"""Are the following item specifications related to the same entity?
    ITEM 1:
    **Title**: {instances.at[i1, 'title']}
    **Authors**: {instances.at[i1, 'authors']}
    **Venue**: {instances.at[i1, 'venue']}
    **Year**: {instances.at[i1, 'year']}

    ITEM 2:
    **Title**: {instances.at[i2, 'title']}
    **Authors**: {instances.at[i2, 'authors']}
    **Venue**: {instances.at[i2, 'venue']}
    **Year**: {instances.at[i2, 'year']}
    """

    chain = ({"question": RunnablePassthrough()} | prompt | llm)
    response = chain.invoke(request)
    return response