import copy
import torch
import torch.nn as nn
from recbole.model.sequential_recommender.sasrec import SASRec


class AdaptorLayer(nn.Module):
    def __init__(self, layers, dropout=0.0):
        super(AdaptorLayer, self).__init__()

        self.layers = layers
        self.dropout = dropout

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
                zip(self.layers[:-1], self.layers[1:])):
            if idx != 0:
                mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
        self.mlp_layers = nn.Sequential(*mlp_modules)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_tensor):
        return self.mlp_layers(input_tensor)


class SASRecText(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.item_embedding = None
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
        self.index_assignment_flag = False

        self.adaptor = AdaptorLayer(
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_item_emb = self.adaptor(self.plm_embedding.weight)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.adaptor(self.plm_embedding.weight)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
