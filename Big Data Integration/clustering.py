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
        n_ctx=512,
        verbose=False,
    )

    return llm

def build_pw_prompt():
    rl_template = """<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant capable of reading and understanding item specifications.
    I will provide you with two item specifications, each containing several attributes. Your task is to determine if they refer to the same entity.

    If you think that the specifications refer to the same entity, respond with "MATCH"; otherwise, respond with "REJECT". Be flexible, as the same entity might be described in different ways: focus on the meaning of the attributes.
    Don't add any additional information to your response.
    Note: If an attribute value is 'nan', consider it as 'not present'.
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
