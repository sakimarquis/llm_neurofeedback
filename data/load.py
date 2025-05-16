import os
import random
import json
from glob import glob
import pandas as pd
from utils import seed_everything


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


def load_dataset(dataset_name, tags, tokenizer):
    seed_everything(42)  # for shuffle dataset; keep unchanged
    if dataset_name == "commonsense":
        dataset = load_commonsense('data/ethics_commonsense', tags, shuffle=False, train_ratio=0.5, tokenizer=tokenizer,
                                   n_sample=1200)
        # do not shuffle commonsense dataset, making a balanced training set
        dataset_name, dataset_label_name = "commonsense", "labels"
    else:
        raise ValueError(f"Unknown dataset {dataset_name}, please choose from happy_sad or commonsense.")
    return dataset, dataset_name, dataset_label_name
