import re
import numpy as np
from utils import load_yaml


def get_tags(model_name):
    try:  # TODO: this is a quick fix, need to find a better way to load the tags
        tag_cfg = load_yaml('./configs/prompt_tags.yml')
    except FileNotFoundError:
        tag_cfg = load_yaml('../configs/prompt_tags.yml')
    if 'llama' in model_name.lower():
        tags = tag_cfg['llama']
    elif 'mistral' in model_name.lower():
        tags = tag_cfg['mistral']
    elif 'gemma' in model_name.lower():
        tags = tag_cfg['gemma']
    elif 'qwen' in model_name.lower():
        tags = tag_cfg['qwen']
    else:
        raise ValueError(f"Model {model_name} not recognized, must be one of 'llama', 'mistral', 'gemma', 'qwen'")
    return tags


def create_system_prompt(text, model_name):
    tags = get_tags(model_name)
    if text is None:
        text = "You are a helpful assistant."
    return f"{tags['start']}{text}{tags['end']}"


def text_to_instruction(text, model_name):
    tags = get_tags(model_name)
    return f"{tags['user']}{text}{tags['assistant']}"


def find_tags_indices(tokens, tags):
    """
    Find occurrences of multiple tags in a single pass through the token sequence.

    Args:
        tokens (List[int]): A sequence of token IDs.
        tags (List[Tuple[str, List[int]]]): A list of tuples where each tuple is
            (tag_name, tag_token_sequence), e.g., ("[INST]", [733, 16289, 28793]).

    Returns:
        occurrences (Dict[str, List[Tuple[int, int]]]): A dictionary where each key is a tag name
            and the value is a list of tuples (start_index, end_index) where the tag was found.
    """
    occurrences = {tag[0]: [] for tag in tags}
    i = 0
    while i < len(tokens):
        match_found = False
        for tag_name, tag_tokens in tags:
            tag_len = len(tag_tokens)
            if tag_len == 0:
                continue
            # Check if there's enough tokens left and if they match the tag sequence.
            if i + tag_len <= len(tokens) and (tokens[i:i + tag_len] == tag_tokens).all():
                occurrences[tag_name].append((i, i + tag_len))
                # Move index past this tag and break to continue with the next token.
                i += tag_len
                match_found = True
                break
        if not match_found:
            i += 1
    return occurrences


def find_tags_indices_batch(tokens_batch, tags):
    """
    Apply the single-pass tag finder to a batch of token sequences.

    Args:
        tokens_batch (List[List[int]]): A batch of tokenized sequences.
        tags (List[Tuple[str, List[int]]]): List of tags to extract, each as (tag_name, tag_token_sequence).

    Returns:
        List[Dict[str, List[Tuple[int, int]]]] : A list of dictionaries where each dictionary contains
            the occurrences of the tags for a single sequence in the batch.
    """
    results = []
    for seq_idx, tokens in enumerate(tokens_batch):
        occ = find_tags_indices(tokens, tags)
        results.append(occ)
    return results


def extract_integer(string):
    try:
        first_number = int(re.search(r'[+-]?\d+', string).group())
    except AttributeError:
        first_number = np.nan
    return first_number
