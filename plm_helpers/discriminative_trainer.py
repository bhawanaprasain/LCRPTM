
    
import torch
import torch.nn.functional as F
from transformers import Trainer


class DiscriminativeTrainer(Trainer):
    def __init__(
        self,
        *args,
        pos_id: int,
        neg_id: int,
        plm_tokenizer=None,
        seq_token_id: int = None,
        label_token_id: int = None,
        class_weights=None,
        margin: float = 0.5,
        margin_weight: float = 0.1,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pos_id = pos_id
        self.neg_id = neg_id
        self.plm_tokenizer = plm_tokenizer
        self.seq_token_id = seq_token_id
        self.label_token_id = label_token_id

        self.class_weights = class_weights
        self.margin = margin
        self.margin_weight = margin_weight
        self.step = 0
        self.label_smoothing = label_smoothing
  

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]


        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        # print(last_hidden.shape)
        logits = outputs.logits

        mask = (labels == self.pos_id) | (labels == self.neg_id)
        has = mask.any(dim=1)
        t = mask.long().argmax(dim=1)
        t_prev = torch.clamp(t - 1, min=0)

        b = torch.arange(logits.size(0), device=logits.device)
        two = logits[b, t_prev][:, [self.neg_id, self.pos_id]]

        true_tok = labels[b, t]
        y = (true_tok == self.pos_id).long()

        if not torch.all(has):
            two = two[has]
            y = y[has]

        weights = (
            self.class_weights.to(two.device)
            if self.class_weights is not None
            else None
        )
    

        ce = F.cross_entropy(two, y, weight=weights, label_smoothing=self.label_smoothing)
        # h = last_hidden[b, t_prev]
        # delta = two[:, 1] - two[:, 0]
        # signed = 2 * y.float() - 1.0
        # signed_delta = signed * delta
        # margin_loss = F.softplus(self.margin - signed_delta).mean()
        # margin_loss = self.margin_weight * margin_loss
        # total = ce +  margin_loss
        # total = ce
        current_epoch = self.state.epoch or 0.0
                
        if current_epoch >= 0.5:

            delta = two[:, 1] - two[:, 0]

            pos_delta = delta[y == 1]
            neg_delta = delta[y == 0]

            if len(pos_delta) > 0:
                pos_anchor = F.softplus(self.margin - pos_delta).mean()
            else:
                pos_anchor = two.new_tensor(0.0)

            if len(neg_delta) > 0:
                neg_anchor = F.softplus(neg_delta + self.margin).mean()
            else:
                neg_anchor = two.new_tensor(0.0)

            anchor_loss = self.margin_weight* pos_anchor* self.class_weights[1] + self.margin_weight* neg_anchor* self.class_weights[0]
            # anchor_loss = self.margin_weight* pos_anchor*  + self.margin_weight* neg_anchor
            
            total = ce  + anchor_loss
            

        else:
            total = ce
  
        if current_epoch >= 0.5 and self.step % 200 ==0 :
                print("CE", ce.item())
                print("Total", total.item())

        self.step += 1
        return (total, outputs) if return_outputs else total

