import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import openai
import json
import time
from claude_util import *
import numpy as np
import random
from tqdm import tqdm
import pandas as pd

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

if __name__ == "__main__":
    """
        run file in main directory: 
            python gsm/gen_gsm.py
    """
    # init Claude Agent
    api_key = list(pd.read_csv('key.csv')['anthropic'])[0]
    claude_agent = Claude(engine='claude-3-haiku-20240307', api_key=api_key)

    agents = 3
    rounds = 2
    random.seed(0)

    generated_description = {}

    questions = read_jsonl("./grade_school_math/data/test.jsonl")
    random.shuffle(questions)

    # cnt = 0
    questions_length = 2 # original: 100
    start_time = time.time()
    for idd in tqdm(range(questions_length)):
        data = questions[idd]
        # gen [0, 2) with claude-3-hiku: input - 5631 tokens, output - 2291 tokens
        # print(f'ask - {cnt}')
        # cnt += 1

        question = data['question']
        answer = data['answer']

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1)
                    agent_context.append(message)
                while True:
                    try:
                        completion = openai.ChatCompletion.create(
                                  model="gpt-3.5-turbo-0613",
                                  messages=agent_context,
                                  proxy='http://127.0.0.1:7890',
                                  temperatue=0.8,
                                  n=1) if claude_agent is None else \
                                    claude_agent.context_ask(agent_context)

                        break
                    except Exception as err:
                        print(err)
                        time.sleep(20)
                        continue

                # assistant_message = construct_assistant_message(completion)
                agent_context.append({
                    "role": "assistant",
                    "content": completion.content[-1].text
                })

        generated_description[question] = (agent_contexts, answer)

    json.dump(generated_description, open("./gsm/gsm_{}_{}.json".format(agents, rounds), "w"))

    # import pdb
    # pdb.set_trace()
    # print(answer)
    # print(agent_context)
    end_time = time.time()
    print(f'request time = {end_time - start_time}') # 25.13s for 2

