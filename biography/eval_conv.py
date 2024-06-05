import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from claude_util import *
import openai
import pandas as pd
import numpy as np
import time
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


def parse_yes_no(string):
    """
    Parses a string containing "yes" or "no" and returns a boolean value.

    Args:
        string (str): The string to parse.

    Returns:
        bool: True if the string contains "yes", False if the string contains "no".

    Raises:
        ValueError: If the input string does not contain "yes" or "no".
    """

    if "uncertain" in string.lower():
        return None
    elif "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None

def filter_people(person):
    people = person.split("(")[0]
    return people

if __name__ == "__main__":
    """
        run file in main directory: 
            python ./biography/eval_conv.py
    """
    # init Claude Agent
    api_key = list(pd.read_csv('key.csv')['anthropic'])[0]
    claude_agent = Claude(engine='claude-3-haiku-20240307', api_key=api_key, max_tokens=600)

    # response of LLM
    response = json.load(open("./biography/biography_3_2.json", "r"))

    # standard answers
    with open("./biography/article.json", "r") as f:
        gt_data = json.load(f)

    gt_data_filter = {}

    for k, v in gt_data.items():
        k = filter_people(k)
        gt_data_filter[k] = v

    gt_data = gt_data_filter

    people = list(response.keys())

    accuracies = []
    start_time = time.time()
    print(f'len(people) = {len(people)}')
    # sys.exit(0)

    for idd in tqdm(range(len(people))):
        person = people[idd]
        if person not in gt_data:
            continue

        gt_description = gt_data[person]
        gt_bullets = parse_bullets(gt_description)
        bio_descriptions = response[person]# [2][-1]['content']


        for description in bio_descriptions:
            bio_description = description[-1]['content']
            # print(f'bio_description = {bio_description}')

            bio_bullets = parse_bullets(bio_description)
            if len(bio_bullets) == 1:
                if len(bio_bullets[0]) < 400:
                    continue

            bio_bullets = " ".join(bio_bullets)
            # continue

            for bullet in gt_bullets:
                message = [{"role": "user", "content": "Consider the following biography of {}: \n {} \n\n Is the above biography above consistent with the fact below? \n\n {} \n Give a single word answer, yes, no, or uncertain. Carefully check the precise dates and locations between the fact and the above biography.".format(person, bio_bullets, bullet)}]
                # valid
                while True:
                    try:
                        completion = openai.ChatCompletion.create(
                                  model="gpt-3.5-turbo-0301",
                                  messages=message,
                                  n=1) if claude_agent is None else \
                                    claude_agent.context_ask(message)
                        break
                    except Exception as err:
                        print(err)
                        time.sleep(20)
                        continue

                # print(message)
                accurate = parse_yes_no(completion["choices"][0]["message"]["content"] if claude_agent is None else completion.content[-1].text)

                if accurate is not None:
                    accuracies.append(float(accurate))


    print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))
    end_time = time.time()
    print(f'request cost time = {end_time - start_time}') # 135.03s for 5 persons
