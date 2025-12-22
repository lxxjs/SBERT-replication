import jittor as jt
import jittor.nn as nn
from SBERT import SBERTModel 

class SiameseSBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased', pooling='mean', num_labels=3):
        super(SiameseSBERT, self).__init__()
        self.sbert = SBERTModel(model_name, pooling)
        self.classifier = nn.Linear(self.sbert.hidden_size * 3, num_labels)
        
    def execute(self, inputs_a, inputs_b, labels=None):
        u = self.sbert(inputs_a['input_ids'], inputs_a['attention_mask'])
        v = self.sbert(inputs_b['input_ids'], inputs_b['attention_mask'])
        
        abs_diff = jt.abs(u - v)
        features = jt.contrib.concat([u, v, abs_diff], dim=1)
        
        logits = self.classifier(features)
        
        if labels is not None:
            loss = nn.cross_entropy_loss(logits, labels)
            return loss, logits
            
        return logits