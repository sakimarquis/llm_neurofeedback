import numpy as np
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm, trange
from analysis.classifiers import PCAClassifier, LogisticRegression, PCAScorer
from typing import Dict, List, Union
from analysis.process_prompts import get_tags, find_tags_indices
from utils import get_model_name


def process_hiddens(hiddens: Dict[int, torch.Tensor], tokenizer=None, data=None, tags=None, method='last_assistant_to_eos_mean', labels=None):
    """Extract the hidden states for the last assistant tag to the first EOS tag and average them
    :return: dict of tensors [batch_size, hidden_size]
    """
    processed_hiddens = {layer: [] for layer in hiddens.keys()}
    new_labels = []
    if method in ['last_assistant_to_eos_mean', 'last_assistant_to_eos_each']:
        assistant_tag_name = tags['assistant']
        assistant_tag = (assistant_tag_name, tokenizer.encode(assistant_tag_name, add_special_tokens=False, return_tensors="pt")[0])
        eos_tag_name = tags['eos']
        eos_tag = (eos_tag_name, tokenizer.encode(eos_tag_name, add_special_tokens=False, return_tensors="pt")[0])
        bos_tag_name = tags['bos']
        bos_tag = (bos_tag_name, tokenizer.encode(bos_tag_name, add_special_tokens=False, return_tensors="pt")[0])

        all_tokens = tokenizer(data, return_tensors="pt", padding=True)['input_ids']
        for sentence_i, tokens in tqdm(enumerate(all_tokens), desc='Processing hiddens'):
            occurrences = find_tags_indices(tokens, [assistant_tag, eos_tag, bos_tag])
            last_assistant_tag_end = occurrences[assistant_tag_name][-1][1]
            sentence_end = None
            if len(occurrences[eos_tag_name]) == 0: # no EOS tag, this happens when no padding
                sentence_end = len(tokens)
            else:
                if tokenizer.padding_side == 'left':
                    # if EOS tag is at the beginning due to left padding, we should ignore them
                    first_bos_tag_start = occurrences[bos_tag_name][0][0]
                    for eos_i in range(len(occurrences[eos_tag_name])):
                        if occurrences[eos_tag_name][eos_i][0] > first_bos_tag_start:
                            sentence_end = occurrences[eos_tag_name][eos_i][0]
                            # this is the first non-padding eos tag
                            break
                    if sentence_end is None:
                        sentence_end = len(tokens) # no non-padding eos tag found
                else:
                    sentence_end = occurrences[eos_tag_name][0][0]
            assert last_assistant_tag_end < sentence_end, "Assistant tag must come before EOS tag"
            # print('Sentence',sentence_i,'from',last_assistant_tag_end,'to',sentence_end,'total',sentence_end-last_assistant_tag_end)
            if labels is not None and method == 'last_assistant_to_eos_each':
                new_labels.extend([labels[sentence_i]] * (sentence_end - last_assistant_tag_end))
            for layer, layer_hiddens in hiddens.items():
                assistant_seq_hiddens = layer_hiddens[sentence_i, last_assistant_tag_end:sentence_end, :]
                if method == 'last_assistant_to_eos_mean':
                    processed_hiddens[layer].append(assistant_seq_hiddens.mean(dim=0))
                elif method == 'last_assistant_to_eos_each':
                    processed_hiddens[layer].extend(list(assistant_seq_hiddens))
                else:
                    raise ValueError(f"Unknown method: {method}, must be one of 'last_assistant_to_eos_mean', 'last_assistant_to_eos_each'")

        for layer, layer_hiddens in processed_hiddens.items():
            processed_hiddens[layer] = torch.stack(layer_hiddens, dim=0).cpu().numpy()

    elif method == 'last':
        for layer, layer_hiddens in tqdm(hiddens.items(), desc="Processing hiddens"):
            processed_hiddens[layer] = layer_hiddens[:, -1, :].cpu().numpy()

    elif method == 'mean':
        for layer, layer_hiddens in tqdm(hiddens.items(), desc="Processing hiddens"):
            processed_hiddens[layer] = layer_hiddens.mean(dim=1).cpu().numpy()

    else:
        raise ValueError(f"Unknown method: {method}, must be one of 'last', 'mean', 'last_assistant_to_eos_mean'")

    if labels is not None and method == 'last_assistant_to_eos_each':
        labels = new_labels
    return processed_hiddens, labels


@torch.inference_mode()
def get_hiddens(model, tokenizer, data: Union[List[str], str], batch_size=16):
    """Extract hidden states from model for given data
    :return: dict of tensors [batch_size, seq_len, hidden_size]
    """
    if isinstance(data, str):
        data = [data]
    logits = []

    if isinstance(model, HookedTransformer):
        collect_loc = "resid_post"
        n_layers = model.cfg.n_layers
        hiddens = {i: [] for i in range(n_layers)}
        tokens = tokenizer(data, return_tensors="pt", padding=True).to(model.cfg.device)['input_ids']
        _, seq_len = tokens.shape
        assert seq_len <= model.cfg.n_ctx, f"Sequence length {seq_len} exceeds model max context length {model.cfg.n_ctx}"

        for i in trange(0, len(tokens), batch_size, desc="Extracting hiddens"):
            # logits: [batch, position, d_vocab]
            batch = tokens[i:i+batch_size]
            logit, cache = model.run_with_cache(batch, names_filter=lambda name: collect_loc in name)
            logits.append(logit.detach().cpu())
            for j in range(n_layers):
                hiddens[j].append(cache[collect_loc, j].detach().cpu())

    else:
        n_layers = model.config.num_hidden_layers
        hiddens = {i: [] for i in range(n_layers)}
        tokens = tokenizer(data, return_tensors="pt", padding=True).to(model.device)
        _, seq_len = tokens['input_ids'].shape
        assert seq_len <= model.config.max_position_embeddings, f"Sequence length {seq_len} exceeds model max context length {model.config.max_position_embeddings}"

        for i in trange(0, len(data), batch_size, desc="Extracting hiddens"):
            batch_tokens = {key: value[i:i + batch_size] for key, value in tokens.items()}
            outputs = model(**batch_tokens, output_hidden_states=True)
            logits.append(outputs.logits.detach().cpu())
            for j in range(n_layers):
                layer_hidden = outputs.hidden_states[j+1].detach().cpu()
                hiddens[j].append(layer_hidden)

    # concatenate all batches
    for j in range(n_layers):
        hiddens[j] = torch.cat(hiddens[j], dim=0)
    logits = torch.cat(logits, dim=0)

    return logits, hiddens


@torch.inference_mode()
def get_hiddens_old(model, tokenizer, data: Union[List[str], str], batch_size=16):
    """Extract hidden states from model for given data
    :return: dict of tensors [batch_size, seq_len, hidden_size]
    """
    if isinstance(data, str) or (isinstance(data, List) and len(data) == 1):
        single_batch = True
    else:
        single_batch = False
    hiddens = {}

    if isinstance(model, HookedTransformer):
        collect_loc = "resid_post"
        n_layers = model.cfg.n_layers
        tokens = tokenizer(data, return_tensors="pt", padding=True).to(model.cfg.device)['input_ids']
        _, seq_len = tokens.shape
        assert seq_len <= model.cfg.n_ctx, f"Sequence length {seq_len} exceeds model max context length {model.cfg.n_ctx}"

        if single_batch:
            # logits: [batch, position, d_vocab]
            logits, cache = model.run_with_cache(tokens, names_filter=lambda name: collect_loc in name)
            logits = logits.detach().cpu()
            for j in range(n_layers):
                hiddens[j] = cache[collect_loc, j].detach().cpu()
        else:
            hiddens = {i: [] for i in range(n_layers)}
            logits = []
            for i in trange(0, len(tokens), batch_size, desc="Extracting hiddens"):
                batch = tokens[i:i+batch_size]
                logit, cache = model.run_with_cache(batch, names_filter=lambda name: collect_loc in name)
                logits.append(logit.detach().cpu())
                for j in range(n_layers):
                    hiddens[j].append(cache[collect_loc, j].detach().cpu())

    else:
        n_layers = model.config.num_hidden_layers
        tokens = tokenizer(data, return_tensors="pt", padding=True).to(model.device)

        if single_batch:
            outputs = model(**tokens, output_hidden_states=True)
            logits = outputs.logits.detach().cpu()
            for j in range(n_layers):
                hiddens[j] = outputs.hidden_states[j+1].detach().cpu()
        else:
            hiddens = {i: [] for i in range(n_layers)}
            logits = []
            for i in trange(0, len(data), batch_size, desc="Extracting hiddens"):
                batch_tokens = {key: value[i:i + batch_size] for key, value in tokens.items()}
                outputs = model(**batch_tokens, output_hidden_states=True)
                logits.append(outputs.logits.detach().cpu())
                for j in range(n_layers):
                    layer_hidden = outputs.hidden_states[j+1].detach().cpu()
                    hiddens[j].append(layer_hidden)

    if not single_batch:
        # concatenate all batches
        for j in range(n_layers):
            hiddens[j] = torch.cat(hiddens[j], dim=0)
        logits = torch.cat(logits, dim=0)

    return logits, hiddens


def train_classify_hiddens(hiddens, labels, method='lr', normalize=True, pc_number=1):
    """Train classifiers for each layer using hidden states
    :param hiddens: dict of tensors [batch_size, hidden_size]
    :param labels: list of labels, [batch_size]
    :param method: str, classifier type, 'lr' for logistic regression, 'pca' for principal component analysis
    :param normalize: bool, whether to normalize the hidden states
    :param pc_number: int, number of principal components to use
    :return: dict of classifiers, key: layer, value: sklearn classifier
    """
    all_classifiers = {}
    all_accuracies = {}
    if method == 'lr':
        for layer, value in tqdm(hiddens.items(), desc="Training classifiers"):
            clf = LogisticRegression(#penalty='l2', C=0.1,
                                     max_iter=1000, solver='saga', n_jobs=-1)
            clf.fit(value, labels, normalize=normalize)
            all_classifiers[layer] = clf
            all_accuracies[layer] = clf.score(value, labels)

    elif method == 'pcadiff':
        for layer, value in tqdm(hiddens.items(), desc="Training classifiers"):
            clf = PCAClassifier()
            clf.fit(value, labels, normalize=normalize)
            all_classifiers[layer] = clf
            all_accuracies[layer] = clf.score(value, labels)
    elif method == 'pcascore':
        for layer, value in tqdm(hiddens.items(), desc="Training classifiers"):
            clf = PCAScorer()
            clf.fit(value, labels, normalize=normalize, pc_number=pc_number)
            all_classifiers[layer] = clf
            all_accuracies[layer] = clf.score(value, labels)
        # directions, means = compute_pc(hiddens)
        # projections = project_hiddens(hiddens, directions, means)
        # signed_directions = {}
        # for layer in hiddens.keys():
        #     direction_sign = compute_signed(projections[layer], labels)
        #     signed_directions[layer] = directions[layer] * direction_sign
        #     pred = (projections[layer] * direction_sign > 0).astype(int)
        #     all_accuracies[layer] = np.mean(np.array(labels) == pred)
        #
        # all_classifiers['directions'] = signed_directions
        # all_classifiers['means'] = means
    else:
        raise ValueError(f"Unknown method: {method}, must be one of 'lr', 'pca'")

    return all_classifiers, all_accuracies

def eval_classify_hiddens(hiddens, labels, all_classifiers, return_type='accuracy'):
    all_scores = {}
    for layer, value in hiddens.items():
        clf = all_classifiers[layer]
        if return_type == 'accuracy':
            score = clf.score(value, labels)
        elif return_type == 'score':
            score = clf.decision_function(value)
        else:
            raise ValueError(f"Unknown return_type: {return_type}, must be one of 'accuracy', 'score'")
        all_scores[layer] = score
    return all_scores


@torch.inference_mode()
def decode_hiddens_score(model, tokenizer, text: str, clfs, layers, tags=None, method='lr'):
    if tags is None:
        tags = get_tags(get_model_name(model))

    logits, hiddens = get_hiddens(model, tokenizer, text)
    processed_hiddens, _ = process_hiddens(hiddens, tokenizer, [text], tags, 'last_assistant_to_eos_mean')

    scores = []

    if isinstance(layers, int):
        return clfs.decision_function(processed_hiddens[layers])[0]

    for layer in layers:
        if method in ['lr', 'pca']:
            score = clfs[layer].decision_function(processed_hiddens[layer])[0]
        # elif method == 'pca':
        #     # score = project_hiddens(hiddens, clfs['directions'], clfs['means'])
        else:
            raise ValueError(f"Unknown method: {method}, must be one of 'lr', 'pca")
        scores.append(score)

    return np.mean(scores)


#
# def compute_pc(hiddens: Dict[int, torch.Tensor]) -> (Dict[int, np.ndarray], Dict[int, torch.Tensor]):
#     """Compute first principal component for each layer of hidden states.
#     hiddens: dict of tensors [batch_size, hidden_size]
#     """
#     pca = PCA(n_components=1)
#     directions = {}
#     means = {}
#     for layer, h in tqdm(hiddens.items(), desc="Computing principal components"):
#         h_mean = torch.mean(h, dim=0, keepdim=True)  # [1, hidden_size]
#         h_centered = h - h_mean
#         pca.fit(h_centered.numpy())
#         directions[layer] = pca.components_.squeeze()
#         means[layer] = h_mean.squeeze()
#     return directions, means
#
#
# def project_hiddens(hiddens, directions, means):
#     projections = {}
#     for layer, h in tqdm(hiddens.items(), desc="Projecting hidden states"):
#         h_centered = h - means[layer][None, :]
#         direction = torch.tensor(directions[layer], dtype=h.dtype, device=h.device)
#         proj = h_centered @ direction
#         proj = proj / torch.norm(proj)
#         projections[layer] = proj.cpu().numpy()
#     return projections
#
#
# def compute_signed(projections: np.ndarray, labels: List[int]):
#     """Compute the correct sign for each layer's PCA direction based on the projections and labels.
#     :param projections: (batch_size, seq_len), PCA projections
#     :param labels: (batch_size,), labels for each instance
#     """
#     assert np.unique(labels).size == 2, "Labels must be binary"
#     assert len(labels) == projections.shape[0], "Number of labels does not match number of instances"
#     labels_np = np.array(labels, dtype=bool)
#     positive = projections[labels_np].mean()
#     negative = projections[~labels_np].mean()
#     direction_sign = np.sign(positive - negative)
#     return direction_sign
