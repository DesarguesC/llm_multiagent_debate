import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glob import glob
import pandas as pd
import json
from claude_util import *
import time
import random
from tqdm import tqdm
import openai

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion, is_claude=False):
    return {"role": "assistant", "content": completion.content[-1].text} if is_claude else \
        {"role": "assistant", "content": completion["choices"][0]["message"]["content"]}


def generate_answer(answer_context, claude_agent=None):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo-0301",
                      messages=answer_context,
                      n=1) if claude_agent is None else \
                        claude_agent.context_ask(answer_context)
        except Exception as err:
            print(err)
            time.sleep(20)
            continue

    return completion


def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer

if __name__ == "__main__":
    """
        run file in main directory: 
            python ./mmlu/gen_mmlu.py
    """
    # init Claude Agent
    api_key = list(pd.read_csv('key.csv')['anthropic'])[0]
    claude_agent = Claude(engine='claude-3-haiku-20240307', api_key=api_key, tokens=1000) # 300, 600 insuficient

    agents = 3
    rounds = 2

    tasks = glob("./mmlu/data/test/*.csv")

    dfs = [pd.read_csv(task) for task in tasks]

    random.seed(0)
    response_dict = {}
    tot_num = 5 # original: 100
    start_time = time.time()
    for i in tqdm(range(tot_num)):
        df = random.choice(dfs)
        ix = len(df)
        idx = random.randint(0, ix-1)

        question, answer = parse_question_answer(df, idx)

        agent_contexts = [[{"role": "user", "content": question}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2 * round - 1)
                    agent_context.append(message)

                completion = generate_answer(agent_context, claude_agent=claude_agent)

                assistant_message = construct_assistant_message(completion, is_claude=True)
                agent_context.append(assistant_message)
                print(completion)

        response_dict[question] = (agent_contexts, answer)

    json.dump(response_dict, open("mmlu_{}_{}.json".format(agents, rounds), "w"))
    end_time = time.time()
    print(f'request cost time = {end_time - start_time}') #