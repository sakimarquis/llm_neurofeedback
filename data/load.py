import os
import random
import json
from glob import glob
import pandas as pd
from utils import seed_everything


def _format_chat_prompt(user_prompt, assistant_response, tags, tokenizer):
    user_tag = tags["user"]
    assistant_tag = tags["assistant"]
    if tokenizer is None:
        return f"{user_tag} {user_prompt} {assistant_tag} {assistant_response}".strip()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _split_examples(examples, train_ratio):
    def pack(split_examples):
        return {
            "data": [item[0] for item in split_examples],
            "labels": [item[1] for item in split_examples],
            "user_data": [item[2] for item in split_examples],
            "assistant_data": [item[3] for item in split_examples],
        }

    if not examples:
        return {"train": pack([]), "test": pack([])}

    split_index = int(len(examples) * train_ratio)
    split_index = max(0, min(split_index, len(examples)))

    train_examples = examples[:split_index]
    test_examples = examples[split_index:]

    print(f"Total examples: {len(examples)}")
    print(f"Train samples: {len(train_examples)}")
    print(f"Test samples: {len(test_examples)}")

    return {"train": pack(train_examples), "test": pack(test_examples)}


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


def happy_sad_dataset(data_path, tags, shuffle=True, train_ratio=0.8, tokenizer=None):
    happiness_path = os.path.join(data_path, "happiness.json")
    sadness_path = os.path.join(data_path, "sadness.json")
    with open(happiness_path, 'r', encoding="utf-8") as f:
        happiness_data = json.load(f)
    with open(sadness_path, 'r', encoding="utf-8") as f:
        sadness_data = json.load(f)

    user_prompt_text = "Say something."
    examples = []

    for sentence in happiness_data:
        assistant_response = sentence.strip() if isinstance(sentence, str) else sentence
        full_prompt = _format_chat_prompt(user_prompt_text, assistant_response, tags, tokenizer)
        examples.append((full_prompt, 1, user_prompt_text, assistant_response))

    for sentence in sadness_data:
        assistant_response = sentence.strip() if isinstance(sentence, str) else sentence
        full_prompt = _format_chat_prompt(user_prompt_text, assistant_response, tags, tokenizer)
        examples.append((full_prompt, 0, user_prompt_text, assistant_response))

    if shuffle:
        random.shuffle(examples)

    return _split_examples(examples, train_ratio)


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
    user_prompt_text = "Say something."

    examples = []
    for _, row in df_all_short.iterrows():
        sentence = row["input"]
        label = row["label"]
        full_prompt = _format_chat_prompt(user_prompt_text, sentence, tags, tokenizer)
        examples.append((full_prompt, label, user_prompt_text, sentence))

    # Shuffle and split
    assert shuffle == False, "Shuffling is not supported for commonsense dataset."
    if shuffle:
        random.shuffle(examples)

    total_examples = len(examples)
    if n_sample is not None:
        examples = examples[:n_sample]
    used_examples = len(examples)

    if total_examples != used_examples:
        print(f"Total examples available: {total_examples}")
    print(f"Used examples: {used_examples}")

    if used_examples == 0:
        return _split_examples(examples, train_ratio)

    base_sample_count = n_sample if n_sample is not None else used_examples
    split_index_raw = int(base_sample_count * train_ratio)
    split_index_capped = max(0, min(split_index_raw, used_examples))
    effective_ratio = split_index_capped / used_examples if used_examples else train_ratio

    return _split_examples(examples, effective_ratio)


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

    user_prompt_text = "Say something."

    examples = []
    for _, row in df_all_short.iterrows():
        sentence = row["statement"]
        label = row["label"]
        full_prompt = _format_chat_prompt(user_prompt_text, sentence, tags, tokenizer)
        examples.append((full_prompt, label, user_prompt_text, sentence))

    total_examples = len(examples)
    if n_sample is not None:
        examples = examples[:n_sample]
    used_examples = len(examples)
    if used_examples == 0:
        return _split_examples(examples, train_ratio)

    if total_examples != used_examples:
        print(f"Total examples available: {total_examples}")
    print(f"Used examples: {used_examples}")

    base_sample_count = n_sample if n_sample is not None else used_examples
    split_index_raw = int(base_sample_count * train_ratio)
    split_index_capped = max(0, min(split_index_raw, used_examples))
    effective_ratio = split_index_capped / used_examples if used_examples else train_ratio

    return _split_examples(examples, effective_ratio)



def load_simple_txt(data_path, tags, tokenizer=None, shuffle=True, train_ratio=0.5, n_sample=1200):
    entries = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                label = int(parts[-1])
            except ValueError:
                continue
            sentence = " ".join(parts[1:-1]).strip()
            if not sentence:
                continue
            entries.append((sentence, label))
    if not entries:
        raise ValueError(f"No usable rows found in {data_path}")
    if shuffle:
        random.shuffle(entries)

    user_prompt_text = "Say something."

    examples = []
    for sentence, label in entries:
        full_prompt = _format_chat_prompt(user_prompt_text, sentence, tags, tokenizer)
        examples.append((full_prompt, label, user_prompt_text, sentence))

    total_examples = len(examples)
    if total_examples < 2:
        raise ValueError("Dataset requires at least two samples for train/test split.")
    if n_sample is not None:
        examples = examples[:n_sample]
    used_examples = len(examples)

    if total_examples != used_examples:
        print(f"Total examples available: {total_examples}")
    print(f"Used examples: {used_examples}")

    if used_examples == 0:
        return _split_examples(examples, train_ratio)

    base_sample_count = n_sample if n_sample is not None else used_examples
    split_index_raw = int(base_sample_count * train_ratio)
    split_index_capped = min(max(split_index_raw, 1), total_examples - 1)
    effective_ratio = split_index_capped / used_examples

    return _split_examples(examples, effective_ratio)


def load_sycophancy_agree(data_path, tags, tokenizer=None, shuffle=True, train_ratio=0.75, n_sample=1200):
    entries = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(' / ')]
            if len(parts) < 3:
                continue
            sentence = parts[1]
            label_part = next((part for part in parts if part.startswith('sycophancy=')), None)
            if label_part is None:
                continue
            try:
                label = int(label_part.split('=', 1)[1])
            except ValueError:
                continue
            entries.append((sentence, label))
    if not entries:
        raise ValueError(f"No usable rows found in {data_path}")
    if shuffle:
        random.shuffle(entries)
    if n_sample is None or n_sample <= 0:
        raise ValueError('n_sample must be a positive integer.')
    total_entries = len(entries)
    use_count = min(n_sample, total_entries)
    selected_entries = entries[:use_count]
    user_prompt_text = 'Respond to the user.'
    examples = []
    for sentence, label in selected_entries:
        full_prompt = _format_chat_prompt(user_prompt_text, sentence, tags, tokenizer)
        examples.append((full_prompt, label, user_prompt_text, sentence))
    used_examples = len(examples)
    if used_examples < 2:
        raise ValueError('Sycophancy agreement dataset requires at least two samples for train/test split.')
    split_index_raw = int(used_examples * train_ratio)
    split_index_capped = min(max(split_index_raw, 1), used_examples - 1)
    effective_ratio = split_index_capped / used_examples
    print(f'Total examples: {total_entries}')
    print(f'Used examples: {used_examples}')
    return _split_examples(examples, effective_ratio)


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
        dataset = happy_sad_dataset("data/emotions", tags=tags, shuffle=True, train_ratio=0.6, tokenizer=tokenizer)
        dataset_name, dataset_label_name = "happy_sad", "labels"
    elif dataset_name == "commonsense":
        dataset = load_commonsense('data/ethics_commonsense', tags, shuffle=False, train_ratio=0.6, tokenizer=tokenizer, n_sample=1500)
        # do not shuffle commonsense dataset, making a balanced training set
        dataset_name, dataset_label_name = "commonsense", "labels"
    # elif dataset_name == "honesty":
    #     dataset = honesty_function_dataset("data/facts_true_false.csv", tags=tags, shuffle=False, n_train=512, include_tf='both')
    #     dataset_name, dataset_label_name = "honesty", "honesty"
    elif dataset_name == "true_false":
        dataset = load_true_false('data/true-false-dataset', tags, shuffle=True, train_ratio=0.6, n_sample=1500)
        dataset_name, dataset_label_name = "true_false", "labels"
    elif dataset_name == "power_seeking":
        dataset = load_simple_txt("data/power-seeking.txt", tags, tokenizer=tokenizer, shuffle=True, train_ratio=0.6, n_sample=1500)
        dataset_name, dataset_label_name = "honesty", "honesty"
    # elif dataset_name == "sycophancy":
    #     dataset = load_simple_txt('data/sycophancy_dataset.txt', tags, tokenizer=tokenizer, shuffle=True, train_ratio=0.5, n_sample=1200)
    #     dataset_name, dataset_label_name = "sycophancy", "labels"
    elif dataset_name == "sycophancy":
        dataset = load_sycophancy_agree('data/sycophancy_agreement.txt', tags, shuffle=True, train_ratio=0.6, n_sample=1500)
        dataset_name, dataset_label_name = "sycophancy_agree", "labels"
    else:
        raise ValueError(f"Unknown dataset {dataset_name}, please choose from happy_sad, commonsense, honesty, true_false, sycophancy, or sycophancy_agree.")

    if len(dataset['test']['data']) != 600:
        print(f"Warning: The test set size is {len(dataset['test']['data'])}, expected 600.")
    return dataset, dataset_label_name


if __name__ == "__main__":
    tags = {"user": "<|user|>", "assistant": "<|assistant|>"}
    # jsonl_path = "../data/power-seeking-inclination.jsonl"
    # loaded_dataset = anthropic_power_seeking(jsonl_path, tags, shuffle=False)
    # dt = pd.DataFrame(loaded_dataset["train"])

    # dt = happy_sad_dataset("../data/emotions", tags=tags, shuffle=True, train_ratio=0.6, tokenizer=None)
    # dt = load_commonsense('../data/ethics_commonsense', tags, shuffle=False, train_ratio=0.6, tokenizer=None, n_sample=1500)
    # dt = load_simple_txt("../data/power-seeking.txt", tags, tokenizer=None, shuffle=True, train_ratio=0.6, n_sample=1500)
    dt = load_true_false('../data/true-false-dataset', tags, shuffle=True, train_ratio=0.6, n_sample=1500)
    # dt = load_sycophancy('../data/sycophancy_dataset.txt', tags, shuffle=True, train_ratio=0.5, n_sample=1200)
    # dt = load_sycophancy_agree('../data/sycophancy_agreement.txt', tags, shuffle=True, train_ratio=0.6, n_sample=1500)
    print(dt["train"].keys())
    print(dt["train"]["data"][:5])
    print(dt["train"]["labels"][:5])
