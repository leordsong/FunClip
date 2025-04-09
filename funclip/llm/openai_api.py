import requests

from openai import OpenAI

from logger import logger
from llm import QWEN_API, DEEPSEEK_API


_MODELS = {
    'deepseek-reasoner': DEEPSEEK_API,
    'deepseek-chat': DEEPSEEK_API,
    'qvq-max': QWEN_API,
    'qwen-vl-max-2025-01-25': QWEN_API,
    'gpt-3.5-turbo': None
}

_REASONING_MODELS = {'deepseek-reasoner', 'qvq-max'}

def reasoning_streaming_decode(completion, include_usage=False):
    """Decode the reasoning stream from the API response"""
    reasoning_content = ""  # Define the complete reasoning process
    answer_content = ""     # Define the complete reply
    is_answering = False   # Determine if the reasoning process has ended and the reply has started

    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            if include_usage:
                logger.info(f'Usage: {chunk.usage}')
        else:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                reasoning_content += delta.reasoning_content
            else:
                # 开始回复
                if delta.content != "" and is_answering is False:
                    is_answering = True
                # 打印回复过程
                answer_content += delta.content

    return answer_content, reasoning_content

def build_single_round_messages(user_prompt, system_prompt=None, base64_image=None):
    messages = []
    if system_prompt is not None and len(system_prompt.strip()):
        messages.append({'role': 'system', 'content': system_prompt})
    if base64_image is not None:
        messages.append({'role': 'user', 'content': [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, # base 64 image string
            },
            {"type": "text", "text": user_prompt},
        ]})
    else:
        messages.append({'role': 'user', 'content': user_prompt})
    return messages

def openai_call(
    apikey,
    model, 
    user_content,
    system_content=None,
    is_json=False,
    image=None,
    temperature=0.0,
):
    assert model in _MODELS, f"Model {model} not supported."
    client = OpenAI(
        # This is the default and can be omitted
        api_key=apikey,
        base_url=_MODELS[model]
    )
    messages = build_single_round_messages(user_content, system_content, image)
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        response_format={
            'type': 'json_object'
        } if is_json else None,
        temperature=temperature,
    )
    
    output = chat_completion.choices[0].message.content
    logger.info("Openai model inference done.")
    return output

def openai_call_reasoning(
    apikey,
    model, 
    user_content,
    system_content=None,
    image=None,
    include_usage=False,
    temperature=0.5,
):
    assert model in _MODELS, f"Model {model} not supported."
    assert model in _REASONING_MODELS, f"Reasoning {model} not supported."
    client = OpenAI(
        # This is the default and can be omitted
        api_key=apikey,
        base_url=_MODELS[model]
    )
    messages = build_single_round_messages(user_content, system_content, image)
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        stream_options={
            "include_usage": True
        } if include_usage else None,
        temperature=temperature,
    )
    
    output, reasoning = reasoning_streaming_decode(chat_completion, include_usage=True)
    logger.info("Openai model inference done.")
    return output, reasoning


def deepseek_call_balance(apikey):
    url = 'https://api.deepseek.com/user/balance'
    # get balance from url
    headers = {'Authorization': f'Bearer {apikey}'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        output = response.json()
        return output['balance_infos'][0]['total_balance']
    else:
        logger.error(f"Failed to fetch balance: {response.status_code}, {response.text}")
        return None