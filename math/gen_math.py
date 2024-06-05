import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import openai
from claude_util import *
import json
import numpy as np
import pandas as pd
import time
import pickle
from tqdm import tqdm

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def generate_answer(answer_context, claude_agent=None):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                              model="gpt-3.5-turbo-0301",
                              messages=answer_context,
                              n=1) if claude_agent is None else \
                            claude_agent.context_ask(answer_context)
            break
        except Exception as err:
            print(err)
            time.sleep(20)
            continue

    return completion


def construct_message(agents, question, idx):

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion, is_claude=False):

    return {"role": "assistant", "content": completion.content[-1].text} if is_claude else \
            {"role": "assistant", "content": completion["choices"][0]["message"]["content"]}

def parse_answer(sentence):
    parts = sentence.split(" ")

    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


if __name__ == "__main__":
    """
        run file in main directory: 
            python math/gen_math.py
    """

    # init Claude Agent
    api_key = list(pd.read_csv('key.csv')['anthropic'])[0]
    claude_agent = Claude(engine='claude-3-haiku-20240307', api_key=api_key)

    answer = parse_answer("My answer is the same as the other agents and AI language model: the result of 12+28*19+6 is 550.")

    agents = 2
    rounds = 3
    np.random.seed(0)

    evaluation_round = 5 # original: 100
    scores = []

    generated_description = {}

    start_time = time.time()
    for round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for agent in range(agents)]

        content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question_prompt, 2*round - 1)
                    agent_context.append(message)

                    # print("message: ", message)

                completion = generate_answer(agent_context, claude_agent=claude_agent)
                # print(completion.content[-1].text)

                assistant_message = construct_assistant_message(completion, is_claude=True)
                agent_context.append({
                    "role": "assistant",
                    "content": completion.content[-1].text
                })


        text_answers = []

        for agent_context in agent_contexts:
            text_answer = string =  agent_context[-1]['content']
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

        generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

        try:
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue

        print("performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))
        # mean = 1.0, var = 0.0

    pickle.dump(generated_description, open("./math/math_agents{}_rounds{}.pq".format(agents, rounds), "wb"))
    end_time = time.time()
    print(f'request cost time = {end_time - start_time}') # cost 54.40s for 5 epoch
    # import pdb
    # pdb.set_trace()
    # print(answer)
    # print(agent_context)

