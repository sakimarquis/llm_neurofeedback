import os
import random
import json
from glob import glob
import pandas as pd
from utils import seed_everything


def tf_statement_dataset(data_path, tags, shuffle=False):
    user_tag = tags['user']
    assistant_tag = tags['assistant']
    df = pd.read_csv(data_path)
    statements = df['statement'].tolist()

    template = "Say some statement about the world."
    examples = []

    for stmt in statements:
        example = f"{user_tag} {template} {assistant_tag} {stmt}".strip()
        examples.append(example)

    if shuffle:
        random.shuffle(examples)
    print(f"Total examples: {len(examples)}")
    return examples


def honesty_function_dataset(data_path, tags, shuffle=False, n_train=512, include_tf='true'):
    # recommend to use include_tf='true' for the honesty function dataset
    # to simplify the task for presentation
    assert include_tf in ['true', 'false', 'both']
    user_tag = tags['user']
    assistant_tag = tags['assistant']
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].tolist()
    false_statements = df[df['label'] == 0]['statement'].tolist()

    template = "Pretend you're {type} person making statements about the world."
    combined = []

    if include_tf in ['true', 'both']:
        for stmt in true_statements:
            honest = f"{user_tag} {template.format(type='an honest')} {assistant_tag} {stmt}".strip()
            dishonest = f"{user_tag} {template.format(type='a dishonest')} {assistant_tag} {stmt}".strip()
            combined.append((honest, dishonest, 1))  # 1 for true

    if include_tf in ['false', 'both']:
        for stmt in false_statements:
            honest = f"{user_tag} {template.format(type='an honest')} {assistant_tag} {stmt}".strip()
            dishonest = f"{user_tag} {template.format(type='a dishonest')} {assistant_tag} {stmt}".strip()
            combined.append((honest, dishonest, 0))  # 0 for false

    if shuffle:
        random.shuffle(combined)

    dataset = {'train': {'data': [], 'labels': [], 'honesty': []}, 'test': {'data': [], 'labels': [], 'honesty': []}}
    for i, pair in enumerate(combined):
        if i < n_train:
            key = 'train'
        else:
            key = 'test'
        dataset[key]['data'] += [pair[0], pair[1]]
        dataset[key]['labels'] += [pair[2], pair[2]]
        dataset[key]['honesty'] += [1, 0]

    print(f"Total pairs: {len(combined)}")
    print(f"Train samples: {len(dataset['train']['data'])}")
    print(f"Test samples: {len(dataset['test']['data'])}")
    return dataset


def simplify_prompt_and_options(user_prompt: str, options: dict, model="gpt-4o-mini") -> tuple:
    """
    Simplifies the user prompt and options such that each is reduced to a single sentence without losing key information.

    Args:
        user_prompt (str): The original question to simplify.
        options (dict): A dictionary of answer options (e.g., {'A': "...", 'B': "..."}).
        model (str): OpenAI model to use.

    Returns:
        tuple: (simplified_user_prompt: str, simplified_options: dict)
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # Set your API key in env variables

    # Build system prompt and user message
    simplification_prompt = (
        "Your task is to shorten text without losing essential meaning. "
        "You will receive a question and multiple choice options. "
        "Each should be rewritten into a single sentence, keeping all key information. "
        "Do not make the content vaguer or change the meaning. Only remove redundancy or simplify phrasing.\n\n"
        f"Question:\n{user_prompt.strip()}\n\n"
        "Options:\n" +
        "\n".join([f"({k}) {v}" for k, v in options.items()])
        + "\n\nReturn your output in JSON format:\n"
        '{"question": "shortened question", "options": {"A": "...", "B": "...", ...}}'
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise text simplifier."},
                {"role": "user", "content": simplification_prompt}
            ],
            temperature=0.3,
            max_tokens=512,
        )
        response_text = response.choices[0].message.content.strip()
        simplified = json.loads(response_text)
        return simplified["question"], simplified["options"]
    except Exception as e:
        print("Error during OpenAI API call:", e)
        return user_prompt, options  # fallback to original


def anthropic_power_seeking(data_path, tags, shuffle=True, train_ratio=0.8, tokenizer=None):
    with open(data_path, 'r') as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    user_tag = tags['user']
    assistant_tag = tags['assistant']

    data = []
    user_data = []
    assistant_data = []
    labels = []

    def sentence2prompt(user_prompt, assistant_response):
        if tokenizer is None:
            return f"{user_tag} {user_prompt} {assistant_tag} {assistant_response}".strip()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return full_prompt

    for item in raw_data:
        question = item["question"]
        # Extract options text from question
        if "Choices:" not in question:
            print(f"Skipping question without choices: {question}")
            continue
        q_part, options_part = question.split("Choices:", 1)
        user_prompt = q_part.replace("Question:", "").replace("\n", " ").strip()
        options_text = options_part.split("Answer:")[0].strip()
        options = {} # option: option_text
        for line in options_text.split("\n"):
            line = line.strip()
            if line.startswith("(") and ")" in line:
                key = line[1] # 'A', 'B', etc.
                value = line[3:].strip()
                options[key] = value
        print(user_prompt, '\n', options)
        for label, answer_key in zip([1, 0], ["answer_matching_behavior", "answer_not_matching_behavior"]):
            answer = item[answer_key] # ' (A)' or ' (B)'
            answer_letter = answer.strip("() ").replace(".", "")
            assistant_response = options.get(answer_letter)
            if assistant_response:
                full_prompt = sentence2prompt(user_prompt, assistant_response)
                data.append(full_prompt)
                user_data.append(user_prompt)
                assistant_data.append(assistant_response)
                labels.append(label)

    # Shuffle and split
    combined = list(zip(data, labels, user_data, assistant_data))
    if shuffle:
        random.shuffle(combined)

    data, labels, user_data, assistant_data = zip(*combined)
    split_index = int(len(data) * train_ratio)

    return {
        "train": {
            "data": list(data[:split_index]),
            "labels": list(labels[:split_index]),
            "user_data": list(user_data[:split_index]),
            "assistant_data": list(assistant_data[:split_index])
        },
        "test": {
            "data": list(data[split_index:]),
            "labels": list(labels[split_index:]),
            "user_data": list(user_data[split_index:]),
            "assistant_data": list(assistant_data[split_index:])
        }
    }



def happy_sad_dataset(data_path, tags, shuffle=True, train_ratio=0.8, tokenizer=None):
    with open(f"{data_path}/happiness.json", 'r') as f:
        happiness_data = json.load(f)
    with open(f"{data_path}/sadness.json", 'r') as f:
        sadness_data = json.load(f)

    # label 1 for happiness and 0 for sadness.
    data = []
    user_data = []
    assistant_data = []
    labels = []

    user_tag = tags['user']
    assistant_tag = tags['assistant']

    def sentence2prompt(sentence):
        if tokenizer is None:
            return f"{user_tag} Say something. {assistant_tag} {sentence}".strip()
        user_prompt = f"Say something."
        response = f"{sentence}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return full_prompt, user_prompt, response


    for sentence in happiness_data:
        full_prompt, user_prompt, response = sentence2prompt(sentence)
        data.append(full_prompt)  # each sentence is wrapped in a list
        user_data.append(user_prompt)
        assistant_data.append(response)
        labels.append(1)
    for sentence in sadness_data:
        full_prompt, user_prompt, response = sentence2prompt(sentence)
        data.append(full_prompt)
        user_data.append(user_prompt)
        assistant_data.append(response)
        labels.append(0)

    # Combine data and labels
    combined = list(zip(data, labels, user_data, assistant_data))
    if shuffle:
        random.shuffle(combined)

    data, labels, user_data, assistant_data = zip(*combined)
    # Split into train and test sets
    split_index = int(len(data) * train_ratio)
    train_data = list(data[:split_index])
    train_labels = list(labels[:split_index])
    train_user_data = list(user_data[:split_index])
    train_assistant_data = list(assistant_data[:split_index])
    test_data = list(data[split_index:])
    test_labels = list(labels[split_index:])
    test_user_data = list(user_data[split_index:])
    test_assistant_data = list(assistant_data[split_index:])
    print(f"Total examples: {len(data)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    return {"train": {"data": train_data, "labels": train_labels, "user_data": train_user_data, "assistant_data": train_assistant_data},
            "test": {"data": test_data, "labels": test_labels, "user_data": test_user_data, "assistant_data": test_assistant_data}}


def load_commonsense(data_path, tags, tokenizer=None, shuffle=False, train_ratio=0.75, n_sample=2000):
    """
    Load the commonsense dataset from the specified path.
    label 0 and 1 are interleaved.
    First part of the dataset is used for training and the second part for testing.
    """
    df_train = pd.read_csv(data_path + '/cm_train.csv')
    df_test = pd.read_csv(data_path + '/cm_test.csv')

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all_short = df_all[df_all['is_short'] == True]
    df_all_short_0 = df_all_short[df_all_short['label'] == 0]
    df_all_short_1 = df_all_short[df_all_short['label'] == 1]
    shorter_len = min(len(df_all_short_0), len(df_all_short_1))
    interleaved_rows = [row for pair in zip(df_all_short_0[:shorter_len].iterrows(), df_all_short_1[:shorter_len].iterrows()) for row in pair]
    df_all_short = pd.DataFrame([row[1] for row in interleaved_rows])

    user_tag = tags["user"]
    assistant_tag = tags["assistant"]
    user_prompt_text = "Say something."

    data = []
    user_data = []
    assistant_data = []
    labels = []

    def format_prompt(sentence):
        if tokenizer is None:
            return f"{user_tag} {user_prompt_text} {assistant_tag} {sentence}".strip()
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt_text},
                {"role": "assistant", "content": sentence},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

    for _, row in df_all_short.iterrows():
        sentence = row["input"]
        label = row["label"]
        full_prompt = format_prompt(sentence)
        data.append(full_prompt)
        user_data.append(user_prompt_text)
        assistant_data.append(sentence)
        labels.append(label)

    # Shuffle and split
    assert shuffle == False, "Shuffling is not supported for commonsense dataset."
    combined = list(zip(data, labels, user_data, assistant_data))
    if shuffle:
        random.shuffle(combined)

    data, labels, user_data, assistant_data = zip(*combined)
    split_index = int(n_sample * train_ratio)

    print(f"Total examples: {len(data)}")
    print(f"Used examples: {n_sample}")
    print(f"Train samples: {len(data[:split_index])}")
    print(f"Test samples: {len(data[split_index:n_sample])}")
    result = {
        "train": {
            "data": list(data[:split_index]),
            "labels": list(labels[:split_index]),
            "user_data": list(user_data[:split_index]),
            "assistant_data": list(assistant_data[:split_index])
        },
        "test": {
            "data": list(data[split_index:n_sample]),
            "labels": list(labels[split_index:n_sample]),
            "user_data": list(user_data[split_index:n_sample]),
            "assistant_data": list(assistant_data[split_index:n_sample])
        }
    }
    return result


def load_true_false(data_path, tags, tokenizer=None, shuffle=True, train_ratio=0.75, n_sample=2000):
    """
    Load the commonsense dataset from the specified path.
    label 0 and 1 are interleaved.
    First part of the dataset is used for training and the second part for testing.
    """
    df_all_short = [pd.read_csv(data_path + '/'+file_name) for file_name in [
        'animals_true_false.csv',
        'cities_true_false.csv',
        'companies_true_false.csv',
        'elements_true_false.csv',
        'facts_true_false.csv',
        'generated_true_false.csv',
        'inventions_true_false.csv',
        ]
    ]
    df_all_short = pd.concat(df_all_short, ignore_index=True)
    if shuffle:
        df_all_short = df_all_short.sample(frac=1).reset_index(drop=True)
    df_all_short_0 = df_all_short[df_all_short['label'] == 0]
    df_all_short_1 = df_all_short[df_all_short['label'] == 1]
    shorter_len = min(len(df_all_short_0), len(df_all_short_1))
    interleaved_rows = [row for pair in zip(df_all_short_0[:shorter_len].iterrows(), df_all_short_1[:shorter_len].iterrows()) for row in pair]
    df_all_short = pd.DataFrame([row[1] for row in interleaved_rows])

    user_tag = tags["user"]
    assistant_tag = tags["assistant"]
    user_prompt_text = "Say something."

    data = []
    user_data = []
    assistant_data = []
    labels = []

    def format_prompt(sentence):
        if tokenizer is None:
            return f"{user_tag} {user_prompt_text} {assistant_tag} {sentence}".strip()
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt_text},
                {"role": "assistant", "content": sentence},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

    for _, row in df_all_short.iterrows():
        sentence = row["statement"]
        label = row["label"]
        full_prompt = format_prompt(sentence)
        data.append(full_prompt)
        user_data.append(user_prompt_text)
        assistant_data.append(sentence)
        labels.append(label)

    combined = list(zip(data, labels, user_data, assistant_data))

    data, labels, user_data, assistant_data = zip(*combined)
    split_index = int(n_sample * train_ratio)

    print(f"Total examples: {len(data)}")
    print(f"Used examples: {n_sample}")
    print(f"Train samples: {len(data[:split_index])}")
    print(f"Test samples: {len(data[split_index:n_sample])}")
    result = {
        "train": {
            "data": list(data[:split_index]),
            "labels": list(labels[:split_index]),
            "user_data": list(user_data[:split_index]),
            "assistant_data": list(assistant_data[:split_index])
        },
        "test": {
            "data": list(data[split_index:n_sample]),
            "labels": list(labels[split_index:n_sample]),
            "user_data": list(user_data[split_index:n_sample]),
            "assistant_data": list(assistant_data[split_index:n_sample])
        }
    }
    return result

def emotion(data_path, shuffle=True):
    data = []
    all_json_files = glob(f"{data_path}/*.json")
    for file in all_json_files:
        with open(file, 'r') as f:
            data.extend(json.load(f))

    if shuffle:
        random.shuffle(data)
    return data


def load_dataset(dataset_name, tags, tokenizer):
    seed_everything(42)  # for shuffle dataset; keep unchanged
    if dataset_name == "happy_sad":
        dataset = happy_sad_dataset("data/emotions", tags=tags, shuffle=True, train_ratio=1.0, tokenizer=tokenizer)
        dataset_name, dataset_label_name = "happy_sad", "labels"
    elif dataset_name == "commonsense":
        dataset = load_commonsense('data/ethics_commonsense', tags, shuffle=False, train_ratio=0.5, tokenizer=tokenizer,
                                   n_sample=1200)
        # do not shuffle commonsense dataset, making a balanced training set
        dataset_name, dataset_label_name = "commonsense", "labels"
    elif dataset_name == "honesty":
        dataset = honesty_function_dataset("data/facts_true_false.csv", tags=tags, shuffle=False, n_train=512, include_tf='both')
        dataset_name, dataset_label_name = "honesty", "honesty"
    elif dataset_name == "true_false":
        dataset = load_true_false('data/true-false-dataset', tags=tags, shuffle=True, train_ratio=0.5, n_sample=1200)
        dataset_name, dataset_label_name = "true_false", "labels"

    else:
        raise ValueError(f"Unknown dataset {dataset_name}, please choose from happy_sad or commonsense.")
    return dataset, dataset_name, dataset_label_name


if __name__ == "__main__":
    tags = {"user": "<|user|>", "assistant": "<|assistant|>"}
    # dataset = happy_sad_dataset("../data/emotions", True)
    # dataset = emotion("../data/emotions", True)
    # print(dataset)

    # jsonl_path = "../data/power-seeking-inclination.jsonl"
    # loaded_dataset = anthropic_power_seeking(jsonl_path, tags, shuffle=False)
    # dt = pd.DataFrame(loaded_dataset["train"])

    # dt = load_commonsense('../data/ethics_commonsense', tags, shuffle=False)
    dt = load_true_false('../data/true-false-dataset', tags, shuffle=True)