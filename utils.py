import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set

def convert_bio_to_spans(bio_tags: List[str]) -> List[Tuple[str, int, int]]:
    """
    Converts BIO tags to entity spans, matching the original paper's implementation.
    Returns list of (entity_type, start_idx, end_idx) tuples.
    """
    spans = []
    current_tag = None
    current_start = None
    
    for i, tag in enumerate(bio_tags):
        if tag.startswith('B-'):
            if current_tag:
                spans.append((current_tag, current_start, i-1))
            current_start = i
            current_tag = tag[2:]  # Remove B- prefix
        elif tag.startswith('I-'):
            if not current_tag:  # Handle malformed I- without B-
                current_start = i
                current_tag = tag[2:]
        else:  # O tag
            if current_tag:
                spans.append((current_tag, current_start, i-1))
                current_tag = None
    
    # Handle entity at end of sequence
    if current_tag:
        spans.append((current_tag, current_start, len(bio_tags)-1))
    
    return spans

def calculate_ner_metrics(true_entities: Set[Tuple], pred_entities: Set[Tuple]) -> Dict:
    """
    Calculates NER metrics exactly as in the original paper.
    Returns precision, recall, and F1 scores.
    """
    # Count correct predictions and total counts
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    # Calculate metrics with same logic as original
    precision = float(nb_correct) / nb_pred if nb_pred > 0 else 0
    recall = float(nb_correct) / nb_true if nb_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': nb_true
    }

def evaluate_relations_trec(true_rels: List[List], pred_rels: List[Dict], nclass: int = 11) -> Tuple:
    """
    Implements the TREC dataset relation evaluation from the original paper.
    """
    # Original paper's relation mapping
    label_map = {
        1: 'Live_In<--', 2: 'OrgBased_In-->',
        3: 'Located_In-->', 4: 'OrgBased_In<--', 
        5: 'Located_In<--', 6: 'Live_In-->',
        7: 'Work_For-->', 8: 'Work_For<--', 
        9: 'Kill-->', 10: 'Kill<--'
    }
    
    rel_map = {
        'Live_In': 0, 'OrgBased_In': 1, 'Located_In': 2,
        'Work_For': 3, 'Kill': 4
    }
    
    # Initialize counters matching original implementation
    relevant = np.zeros(nclass - 1)
    correct = np.zeros(nclass - 1)
    predicted = np.zeros(nclass - 1)
    
    for true_instance, pred_instance in zip(true_rels, pred_rels):
        for true_rel in true_instance:
            if true_rel[0] != 0:  # Skip 'O' relations
                category = label_map[true_rel[0]][:-3]
                relevant[rel_map[category]] += 1
                
        # Count predictions
        for key, pred_label in pred_instance.items():
            if pred_label != 0:
                category = label_map[pred_label][:-3]
                predicted[rel_map[category]] += 1
                
                # Check if prediction is correct
                label, start, end, batch_idx = key
                if (pred_label, list(start), list(end)) in true_instance:
                    correct[rel_map[category]] += 1

    # Calculate metrics
    precision = correct.sum() / predicted.sum() if predicted.sum() > 0 else 0
    recall = correct.sum() / relevant.sum() if relevant.sum() > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # Generate detailed report
    report = generate_relation_report(correct, predicted, relevant, rel_map)

    return precision, recall, f1, report

def generate_relation_report(correct: np.ndarray, predicted: np.ndarray, 
                           relevant: np.ndarray, rel_map: Dict) -> str:
    """
    Generates a detailed classification report for relations,
    matching the format from the original paper.
    """
    headers = ["precision", "recall", "f1-score", "support"]
    name_width = max(len(name) for name in rel_map.keys())
    width = max(name_width, len("avg / total"))
    
    # Header
    report = "{:>{width}} {:>9} {:>9} {:>9} {:>9}\n".format(
        "", *headers, width=width)
    report += "\n"
    
    # Per-class metrics
    ps, rs, f1s, s = [], [], [], []
    for rel_type, idx in rel_map.items():
        p = correct[idx] / predicted[idx] if predicted[idx] > 0 else 0
        r = correct[idx] / relevant[idx] if relevant[idx] > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        
        report += "{:>{width}} {:>9.4f} {:>9.4f} {:>9.4f} {:>9d}\n".format(
            rel_type, p, r, f1, int(relevant[idx]), width=width)
            
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(relevant[idx])
    
    # Average metrics
    report += "\n"
    report += "{:>{width}} {:>9.4f} {:>9.4f} {:>9.4f} {:>9d}\n".format(
        "avg / total",
        np.average(ps, weights=s),
        np.average(rs, weights=s),
        np.average(f1s, weights=s),
        int(sum(s)),
        width=width
    )
    
    return report

def create_training_examples(tokens: List[str], bio_tags: List[str], 
                           relations: List[Tuple]) -> Dict:
    """
    Creates spaCy-compatible training examples from the original data format.
    """
    text = " ".join(tokens)
    spans = convert_bio_to_spans(bio_tags)
    
    # Calculate character offsets
    char_offsets = []
    current_offset = 0
    for token in tokens:
        char_offsets.append((current_offset, current_offset + len(token)))
        current_offset += len(token) + 1  # +1 for space
        
    # Convert spans to character offsets
    entities = []
    for ent_type, start_idx, end_idx in spans:
        start_char = char_offsets[start_idx][0]
        end_char = char_offsets[end_idx][1]
        entities.append({
            "start": start_char,
            "end": end_char,
            "label": ent_type
        })
    
    # Convert relations to character-based format
    relations_formatted = []
    for rel_type, ent1_indices, ent2_indices in relations:
        if rel_type == 0:  # Skip 'O' relations
            continue
        relations_formatted.append({
            "head": (char_offsets[min(ent1_indices)][0], 
                    char_offsets[max(ent1_indices)][1]),
            "tail": (char_offsets[min(ent2_indices)][0], 
                    char_offsets[max(ent2_indices)][1]),
            "type": rel_type
        })
    
    return {
        "text": text,
        "entities": entities,
        "relations": relations_formatted
    }