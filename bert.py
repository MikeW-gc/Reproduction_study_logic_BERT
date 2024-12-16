# bert.py
import spacy
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertTokenizerFast
import pickle
import numpy as np
import torch.nn.functional as F  
from tqdm import tqdm
import os
import evaluate as eva

class BertNERRE(nn.Module):
    def __init__(self, num_ner_labels=9, num_rel_labels=11):
        super().__init__()
        # Load BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # NER head
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, num_ner_labels)
        
        # RE head
        rel_input_size = self.bert.config.hidden_size * 2  # concatenated entity pairs
        self.rel_classifier = nn.Sequential(
            nn.Linear(rel_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_rel_labels)
        )

        # Logic components
        self.linear_con = nn.Linear(1, 1).cuda()
        self.linear_dis = nn.Linear(1, 1).cuda()
        self.weights = nn.Parameter(torch.ones(1, 15)).cuda()
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        ner_logits = self.ner_classifier(sequence_output)
        
        return ner_logits, sequence_output

    def predict_relations(self, sequence_output, entity_spans):
        """
        Predict relations between entity pairs
        
        Args:
            sequence_output: BERT sequence output
            entity_spans: List of entity span indices
        Returns:
            Tensor of relation logits for each entity pair
        """
        rel_logits = []
        
        # For each possible entity pair
        for i, e1 in enumerate(entity_spans):
            for j, e2 in enumerate(entity_spans[i+1:], i+1):
                # Get entity representations by averaging their token embeddings
                e1_repr = sequence_output[0, e1, :].mean(dim=0)
                e2_repr = sequence_output[0, e2, :].mean(dim=0)
                
                # Concatenate entity pair representations
                pair_repr = torch.cat([e1_repr, e2_repr])
                
                # Get relation logits
                logits = self.rel_classifier(pair_repr)
                rel_logits.append(logits)
        
        return torch.stack(rel_logits) if rel_logits else torch.zeros((0, 11)).to(sequence_output.device)

    def compute_logic_loss(self, output_chunk, out_rel, label_map):
        """Compute logic loss based on relationship rules"""
        loss = 0.0
        count = 0
        weights = torch.sigmoid(self.weights)
        
        for key in out_rel.keys():
            rel_vec = out_rel[key]
            rel = torch.argmax(rel_vec).item()
            (l, s, e, batch_idx) = key
            out_s = output_chunk[batch_idx][list(s)]
            out_e = output_chunk[batch_idx][list(e)]

            try:
                # Rule 1 & 2: Live_In relations
                if rel in [6, 1]:  # Live_In--> or Live_In<--
                    if torch.argmax(out_e[0]) == 7:  # Location
                        loss += weights[0,0] * ((out_s[0, 3] - torch.sigmoid(self.linear_con((rel_vec[rel] + out_e[0, 7] - 2).view(-1, 1)))) ** 2)
                        count += 1
                    if torch.argmax(out_s[0]) == 3:  # Person
                        loss += weights[0,1] * ((out_e[0, 7] - torch.sigmoid(self.linear_con((rel_vec[rel] + out_s[0, 3] - 2).view(-1, 1)))) ** 2)
                        count += 1
                
                # Rule 3 & 4: OrgBased_In relations
                elif rel in [2, 4]:  # OrgBased_In--> or OrgBased_In<--
                    if torch.argmax(out_e[0]) == 7:  # Location
                        loss += weights[0,2] * ((out_s[0, 5] - torch.sigmoid(self.linear_con((rel_vec[rel] + out_e[0, 7] - 2).view(-1, 1)))) ** 2)
                        count += 1
                    if torch.argmax(out_s[0]) == 5:  # Organization
                        loss += weights[0,3] * ((out_e[0, 7] - torch.sigmoid(self.linear_con((rel_vec[rel] + out_s[0, 5] - 2).view(-1, 1)))) ** 2)
                        count += 1
                
                # Rule 5: Located_In relations
                elif rel in [3, 5]:  # Located_In--> or Located_In<--
                    if torch.argmax(out_e[0]) == 7:  # Location
                        loss += weights[0,4] * ((out_s[0, 7] - torch.sigmoid(self.linear_con((rel_vec[rel] + out_e[0, 7] - 2).view(-1, 1)))) ** 2)
                        count += 1
                
                # Rule 6: Kill relations
                elif rel in [9, 10]:  # Kill--> or Kill<--
                    if torch.argmax(out_e[0]) == 3:  # Person
                        loss += weights[0,5] * ((out_s[0, 3] - torch.sigmoid(self.linear_con((rel_vec[rel] + out_e[0, 3] - 2).view(-1, 1)))) ** 2)
                        count += 1
                
                # Rule 7 & 8: Work_For relations
                elif rel in [7, 8]:  # Work_For--> or Work_For<--
                    if torch.argmax(out_e[0]) == 5:  # Organization
                        loss += weights[0,6] * ((out_s[0, 3] - torch.sigmoid(self.linear_con((rel_vec[rel] + out_e[0, 5] - 2).view(-1, 1)))) ** 2)
                        count += 1
                    if torch.argmax(out_s[0]) == 3:  # Person
                        loss += weights[0,7] * ((out_e[0, 5] - torch.sigmoid(self.linear_con((rel_vec[rel] + out_s[0, 3] - 2).view(-1, 1)))) ** 2)
                        count += 1
            
            except IndexError:
                continue

        return loss / count if count > 0 else torch.tensor(0.0).cuda()
    
class JointNERREPipeline:
    def __init__(self):
        # Initialize BERT-based model
        self.model = BertNERRE().cuda()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        # Load data
        self.train_words = self.load_pickle("data/words.train")
        self.test_words = self.load_pickle("data/words.test")
        self.train_labels_chunk = self.load_pickle("data/labels_chunk.train")[0]
        self.test_labels_chunk = self.load_pickle("data/labels_chunk.test")[0]
        self.train_labels_rel = self.load_pickle("data/labels_2rel.train")
        self.test_labels_rel = self.load_pickle("data/labels_2rel.test")
        
        # Setup optimizers
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        # Label mappings
        self.label_map = {0:'O', 1:'B-Other', 2:'I-Other', 3:'B-Peop', 
                         4:'I-Peop', 5:'B-Org', 6:'I-Org', 7:'B-Loc', 8:'I-Loc'}
        self.entity_map = {'Other':0, 'Peop':1, 'Org':2, 'Loc':3}
        self.rel_map = {'Live_In':0, 'OrgBased_In':1, 'Located_In':2, 'Work_For':3, 'Kill':4}
        self.label_rel = {
            0: 'O',
            1: 'Live_In<--', 
            2: 'OrgBased_In-->', 
            3: 'Located_In-->', 
            4: 'OrgBased_In<--',
            5: 'Located_In<--',
            6: 'Live_In-->',
            7: 'Work_For-->',
            8: 'Work_For<--',
            9: 'Kill-->',
            10: 'Kill<--'
        }

    def train(self, n_epochs=150, checkpoint_path=None, save_freq=1, model_name="bertnologic"):
        # Load checkpoint if provided
        start_epoch = self.load_checkpoint(checkpoint_path, model_name) if checkpoint_path else 0
        
        self.model.train()
        best_f1 = 0.0  # Track best combined F1 score
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        for epoch in range(start_epoch, start_epoch + n_epochs):
            total_loss = 0
            
            # Training step
            for words, ner_labels, rel_labels in zip(
                self.train_words, self.train_labels_chunk, self.train_labels_rel
            ):
                # Prepare inputs
                inputs, aligned_labels = self.tokenize_and_align_labels(words, ner_labels)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                aligned_labels = aligned_labels.cuda()
                
                # Forward pass
                self.optimizer.zero_grad()
                ner_logits, sequence_output = self.model(**inputs)
                
                # Calculate NER loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                ner_loss = loss_fct(ner_logits.view(-1, len(self.label_map)), 
                                  aligned_labels.view(-1))

                # Get NER predictions for entity spans
                predictions = torch.argmax(ner_logits, dim=-1)
                word_ids = self.tokenizer(words, is_split_into_words=True, return_tensors="pt").word_ids()
                word_preds = []
                last_word_id = None
                for pred, word_id in zip(predictions[0], word_ids):
                    if word_id is not None and word_id != last_word_id:
                        pred_tag = self.label_map[pred.item()]
                        word_preds.append(pred_tag)
                        last_word_id = word_id
                
                entity_spans, _ = self.get_entity_spans_and_labels(word_preds)

                # Relation prediction and loss
                rel_logits = self.model.predict_relations(sequence_output, entity_spans)
                rel_labels = self.prepare_relation_labels(rel_labels, entity_spans)

                if len(rel_logits)>0:
                    rel_loss = loss_fct(rel_logits, rel_labels)
                    loss = ner_loss + rel_loss
                else:
                    loss = ner_loss

           
                # Backward pass
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            # Evaluate after each epoch
            ner_metrics, rel_metrics = self.evaluate(epoch)

            ner_metrics, rel_metrics = self.evaluate(epoch)
        
            # Save results for this epoch
            self.save_epoch_results(epoch, (ner_metrics, rel_metrics), model_name, is_training=True)
            
            # Print metrics
            print(f"Epoch {epoch}")
            print(f"Loss: {total_loss:.4f}")
            print(f"Entity extraction performance:")
            print(f"precision: {ner_metrics['precision']:.4f}\t"
                  f"recall: {ner_metrics['recall']:.4f}\t"
                  f"f1: {ner_metrics['f1']:.4f}")
            print(ner_metrics['report'])
            
            print(f"\nRelation extraction performance:")
            print(f"precision: {rel_metrics['precision']:.4f}\t"
                  f"recall: {rel_metrics['recall']:.4f}\t"
                  f"f1: {rel_metrics['f1']:.4f}")
            print(rel_metrics['report'])

            # Save checkpoints
            combined_f1 = (ner_metrics['f1'] + rel_metrics['f1']) / 2
            if epoch % save_freq == 0:
                self.save_checkpoint(epoch, (ner_metrics, rel_metrics), model_name)
            
            if combined_f1 > best_f1:
                best_f1 = combined_f1
                best_checkpoint_dir = os.path.join(script_dir, "checkpoints", model_name, "best")
                self.save_checkpoint(epoch, (ner_metrics, rel_metrics), 
                                checkpoint_dir=best_checkpoint_dir)
                
    def evaluate(self, epoch=None):
        self.model.eval()
        predicted_labels = []
        predicted_relations = []

        print("\nStarting evaluation...")

        true_labels = []
        for sent_labels in self.test_labels_chunk:
            sent_strings = [self.label_map[label] for label in sent_labels]
            true_labels.append(sent_strings)
        
        with torch.no_grad():
            for idx, words in enumerate(self.test_words):
                print(f"\nProcessing test instance {idx}")
                tokenized_inputs = self.tokenize_and_align_labels(words)
                inputs = {k: v.cuda() for k, v in tokenized_inputs.items()}
                ner_logits, sequence_output = self.model(**inputs)

                predictions = torch.argmax(ner_logits, dim=-1)
                print(f"Raw predictions shape: {predictions.shape}")
                
                # NER predictions
                word_ids = self.tokenizer(words, is_split_into_words=True, return_tensors="pt").word_ids()
                word_preds = []
                last_word_id = None

                for pred, word_id in zip(predictions[0], word_ids):
                    if word_id is not None and word_id != last_word_id:
                        pred_tag = self.label_map[pred.item()]
                        word_preds.append(pred_tag)
                        last_word_id = word_id
                
                print(f"Converted predictions: {word_preds[:5]}...")
                predicted_labels.append(word_preds)

                # Relation predictions
                entity_spans, entity_types = self.get_entity_spans_and_labels(word_preds)
                
                # Get relation predictions
                rel_logits = self.model.predict_relations(sequence_output, entity_spans)
                rel_preds = torch.argmax(rel_logits, dim=-1)
                rel_dict = {}
                pred_idx = 0

                for i, e1 in enumerate(entity_spans):
                    for j, e2 in enumerate(entity_spans[i+1:], i+1):
                        if pred_idx < len(rel_preds):
                            key = (0, tuple(e1), tuple(e2), idx)
                            rel_dict[key] = rel_preds[pred_idx].item()
                            pred_idx += 1

                predicted_relations.append(rel_dict)

            # Calculate metrics
            ner_metrics = eva.f1_score(true_labels, predicted_labels)
            
            p, r, f1, report = eva.report_relation_trec(
                self.test_labels_rel,
                predicted_relations,
                6  # Fixed number: 5 relation types + 'O'
            )

            # Save predictions if epoch is provided
            if epoch is not None:
                self.save_predictions(epoch, predicted_labels, predicted_relations)

            return {
                'precision': eva.precision_score(true_labels, predicted_labels),
                'recall': eva.recall_score(true_labels, predicted_labels),
                'f1': ner_metrics,
                'report': eva.classification_report(true_labels, predicted_labels)
            }, {
                'precision': p,
                'recall': r,
                'f1': f1,
                'report': report
            }
    def tokenize_and_align_labels(self, words, labels=None):
        """Tokenize words and align labels with BERT wordpieces"""
        tokenized_inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        if labels is None:
            return tokenized_inputs
            
        word_ids = tokenized_inputs.word_ids()
        aligned_labels = [-100] * len(word_ids)  # -100 is ignored in loss calculation
        
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                aligned_labels[idx] = labels[word_id]
                
        return tokenized_inputs, torch.tensor(aligned_labels)
        
    def get_entity_spans_and_labels(self, word_preds):
        """Extract entity spans and their types from predictions"""
        entity_spans = []
        entity_types = []
        current_span = []
        current_type = None
        
        for i, label in enumerate(word_preds):
            if label.startswith('B-'):
                if current_span:
                    entity_spans.append(current_span)
                    entity_types.append(current_type)
                current_span = [i]
                current_type = label[2:]  # Get entity type without B-/I-
            elif label.startswith('I-'):
                if current_span:  # Only add to span if we have a B- already
                    current_span.append(i)
            elif current_span:  # End of span
                entity_spans.append(current_span)
                entity_types.append(current_type)
                current_span = []
                current_type = None
                
        if current_span:  # Handle last span
            entity_spans.append(current_span)
            entity_types.append(current_type)
            
        return entity_spans, entity_types

    def prepare_relation_labels(self, rel_data, entity_spans):
        labels = []
        
        # Handle list format (training data)
        if isinstance(rel_data, list):
            # Convert the relation data list into a dictionary format
            rel_dict = {}
            for rel in rel_data:
                # rel format is (relation_type, [e1_start, e1_end], [e2_start, e2_end])
                key = (0, tuple(rel[1]), tuple(rel[2]), 0)  # Convert list indices to tuples
                rel_dict[key] = rel[0]  # relation type
        
        # Create labels for each possible entity pair
        for i, e1 in enumerate(entity_spans):
            for j, e2 in enumerate(entity_spans[i+1:], i+1):
                key = (0, tuple(e1), tuple(e2), 0)
                label = rel_dict.get(key, 0)  # Default to 0 (no relation)
                labels.append(label)
                
        return torch.tensor(labels, device='cuda') if labels else torch.zeros(0, device='cuda')
    



    def verify_data(self):
        """Verify data formats and sizes"""
        print(f"Training data size: {len(self.train_words)}")
        print(f"Test data size: {len(self.test_words)}")
        print("\nSample training instance:")
        idx = 0
        print(f"Words: {self.train_words[idx]}")
        print(f"NER labels: {[self.label_map[l] for l in self.train_labels_chunk[idx]]}")
        print(f"Relations: {self.train_labels_rel[idx]}")\
        
    def debug_relation_data(self):
        print("Sample relation data:")
        print("\nFirst training instance:")
        print("Words:", self.train_words[0])
        print("Type of rel_labels:", type(self.train_labels_rel[0]))
        print("Content of rel_labels:", self.train_labels_rel[0])

    def debug_predictions(self):
        """Add this method to JointNERREPipeline to debug prediction format"""
        # Get first test instance
        words = self.test_words[0]
        true_rels = self.test_labels_rel[0]
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenize_and_align_labels(words)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            ner_logits, sequence_output = self.model(**inputs)
            
            # Show predicted format
            print("True relations format:")
            print(true_rels)
            
            # Show our prediction format 
            rel_pred = {}
            for i in range(len(words)):
                for j in range(len(words)):
                    if i < j:
                        rel_pred[(0, (i,), (j,), 0)] = 0
            print("\nOur prediction format:")
            print(rel_pred)

    def save_checkpoint(self, epoch, metrics, model_name="bertnologic", checkpoint_dir=None):
        """Save model checkpoint with error handling"""
        try:
            # Create checkpoint directory if it doesn't exist
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if checkpoint_dir is None:
                checkpoint_dir = os.path.join(script_dir, "checkpoints", model_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Create checkpoint name
            ner_f1 = metrics[0]['f1']
            rel_f1 = metrics[1]['f1']
            checkpoint_name = f"checkpoint_epoch_{epoch}_ner_{ner_f1:.3f}_rel_{rel_f1:.3f}"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
            # Prepare checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'ner_metrics': metrics[0],
                'rel_metrics': metrics[1],
            }
            
            # Save with temporary file
            temp_path = f"{checkpoint_path}.temp.pt"
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, f"{checkpoint_path}.pt")
            
            print(f"\nCheckpoint saved: {checkpoint_path}.pt")
        
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {str(e)}")
            # Continue training even if saving fails
            pass

    def load_checkpoint(self, checkpoint_path=None, model_name="bertnologic"):
        """Load model checkpoint and return the starting epoch"""
        if checkpoint_path is None:
            # Get script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_dir = os.path.join(script_dir, "checkpoints", model_name)
            
            if not os.path.exists(checkpoint_dir):
                print(f"No checkpoints directory found for model {model_name}")
                return 0
                
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if not checkpoints:
                print(f"No checkpoints found for model {model_name}")
                return 0
                
            # Sort by epoch number
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[2]))[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        # Now check the final checkpoint_path
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return 0
            
        checkpoint = torch.load(checkpoint_path)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"NER F1: {checkpoint['ner_metrics']['f1']:.4f}")
        print(f"Relation F1: {checkpoint['rel_metrics']['f1']:.4f}")
        
        return checkpoint['epoch'] + 1
    
    def save_epoch_results(self, epoch, metrics, model_name="bertnologic", is_training=True):
        """
        Save detailed epoch results to files organized by model and metric type
        
        Args:
            epoch: Current epoch number
            metrics: Tuple of (ner_metrics, rel_metrics) 
            model_name: Name of model variant
            is_training: Whether these are training or validation results
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results", model_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Separate files for different metric types
        phase = "train" if is_training else "val"
        
        # Save NER metrics
        ner_metrics = metrics[0]
        ner_file = os.path.join(results_dir, f"ner_{phase}_metrics.csv")
        
        # Create header if file doesn't exist
        if not os.path.exists(ner_file):
            with open(ner_file, 'w') as f:
                f.write("epoch,precision,recall,f1\n")
                
        # Append metrics
        with open(ner_file, 'a') as f:
            f.write(f"{epoch},{ner_metrics['precision']},{ner_metrics['recall']},{ner_metrics['f1']}\n")
            
        # Save detailed NER classification report
        ner_report_file = os.path.join(results_dir, f"ner_{phase}_report_epoch_{epoch}.txt")
        with open(ner_report_file, 'w') as f:
            f.write(ner_metrics['report'])
        
        # Save RE metrics
        rel_metrics = metrics[1]
        rel_file = os.path.join(results_dir, f"rel_{phase}_metrics.csv")
        
        # Create header if file doesn't exist
        if not os.path.exists(rel_file):
            with open(rel_file, 'w') as f:
                f.write("epoch,precision,recall,f1\n")
                
        # Append metrics
        with open(rel_file, 'a') as f:
            f.write(f"{epoch},{rel_metrics['precision']},{rel_metrics['recall']},{rel_metrics['f1']}\n")
            
        # Save detailed RE report
        rel_report_file = os.path.join(results_dir, f"rel_{phase}_report_epoch_{epoch}.txt")
        with open(rel_report_file, 'w') as f:
            f.write(rel_metrics['report'])
        
    def save_predictions(self, epoch, predicted_labels, predicted_relations, model_name="bertnologic"):
        """
        Save model predictions for each epoch
        
        Args:
            epoch: Current epoch number
            predicted_labels: List of predicted NER labels
            predicted_relations: List of predicted relations
            model_name: Name of model variant
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        predictions_dir = os.path.join(script_dir, "predictions", model_name, f"epoch_{epoch}")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save NER predictions
        ner_file = os.path.join(predictions_dir, "ner_predictions.txt")
        with open(ner_file, 'w') as f:
            for sent_labels in predicted_labels:
                f.write(' '.join(sent_labels) + '\n')
        
        # Save relation predictions
        rel_file = os.path.join(predictions_dir, "rel_predictions.pkl")
        with open(rel_file, 'wb') as f:
            pickle.dump(predicted_relations, f)

    def debug_relation_data(self):
        print("Sample relation data:")
        for i in range(3):  # Look at first 3 examples
            print(f"\nSample {i}:")
            print("Words:", self.train_words[i])
            print("Relations:", self.train_labels_rel[i])
            # Count non-zero relations
            if isinstance(self.train_labels_rel[i], dict):
                non_zero = sum(1 for v in self.train_labels_rel[i].values() if v != 0)
                print(f"Number of non-zero relations: {non_zero}")

    def debug_relation_prediction(self):
        self.model.eval()
        with torch.no_grad():
            # Get first training example
            words = self.train_words[0]
            true_rels = self.train_labels_rel[0]
            
            # Get model outputs
            inputs = self.tokenize_and_align_labels(words)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            ner_logits, sequence_output = self.model(**inputs)
            
            # Print sequence output shape
            print("Sequence output shape:", sequence_output.shape)
            
            # Get entity spans
            predictions = torch.argmax(ner_logits, dim=-1)
            word_ids = self.tokenizer(words, is_split_into_words=True, return_tensors="pt").word_ids()
            
            # Print predicted entities
            print("\nPredicted entities:")
            entity_spans = []
            current_span = []
            word_preds = []
            last_word_id = None
            
            for pred, word_id in zip(predictions[0], word_ids):
                if word_id is not None and word_id != last_word_id:
                    pred_tag = self.label_map[pred.item()]
                    word_preds.append(pred_tag)
                    last_word_id = word_id
                    
            for i, label in enumerate(word_preds):
                if label.startswith('B-'):
                    if current_span:
                        entity_spans.append(current_span)
                    current_span = [i]
                elif label.startswith('I-'):
                    if current_span:
                        current_span.append(i)
                elif current_span:
                    entity_spans.append(current_span)
                    current_span = []
            if current_span:
                entity_spans.append(current_span)
                
            print("Entity spans:", entity_spans)
            print("True relations:", true_rels)

    @staticmethod
    def load_pickle(path):
            """Load data from pickle file"""
            with open(path, 'rb') as f:
                return pickle.load(f)

if __name__ == "__main__":
    pipeline = JointNERREPipeline()
    # pipeline.verify_data()  
    # pipeline.debug_predictions()
    print("Pipeline initialized")
    print(f"Training data size: {len(pipeline.train_words)}")
    print("Starting training...")
    pipeline.debug_relation_data()
    pipeline.debug_relation_prediction()
    pipeline.train(n_epochs=150, save_freq=5)  # Save checkpoint every 5 epochs
