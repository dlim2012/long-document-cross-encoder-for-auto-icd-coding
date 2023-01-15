import torch
from evaluation import eval_func

from transformers import AutoConfig, AutoModel, AutoTokenizer

class MORES(torch.nn.Module):
    def __init__(self, args, device, chunk_notes):
        super(MORES, self).__init__()

        config = AutoConfig.from_pretrained(args.config_name)
        self.encoder = AutoModel.from_pretrained(args.config_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(args.config_name)

        if args.loss == 'cross_entropy':
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        elif args.loss == 'bce':
            self.sigmoid = torch.nn.Sigmoid()
            self.bce_loss = torch.nn.BCELoss(reduction='mean')
        else:
            raise ValueError()

        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=args.vector_dim, num_heads=1, batch_first=True)
        self.layerNorm1 = torch.nn.LayerNorm(args.vector_dim)

        self.self_attention = torch.nn.MultiheadAttention(embed_dim=args.vector_dim, num_heads=1, batch_first=True)
        self.layerNorm2 = torch.nn.LayerNorm(args.vector_dim)
        self.linear = torch.nn.Linear(args.vector_dim, 1)

        self.eval = eval_func

        self.args = args
        self.device = device
        self.loss_type = args.loss
        self.switch_qd = args.switch_qd
        self.chunk_notes = chunk_notes
        self.use_self_attention = args.self_attention

        if not self.chunk_notes:
            raise NotImplementedError()

    def interaction_block(self, qry_reps, doc_reps):
        if self.switch_qd:
            qry_bsz = qry_reps.shape[0]
            doc_reps = doc_reps.reshape((-1, doc_reps.shape[2]))  # (doc length, vector_dim)

            doc_reps = torch.repeat_interleave(doc_reps.unsqueeze(0), qry_bsz,
                                               dim=0)  # (qry_bsz, doc_len, vector_dim)
        else:
            doc_bsz = doc_reps.shape[0]
            qry_reps = qry_reps.reshape((-1, qry_reps.shape[2]))  # (qry length, vector_dim)

            qry_reps = torch.repeat_interleave(qry_reps.unsqueeze(0), doc_bsz,
                                               dim=0)  # (doc_bsz, qry_len, vector_dim)


        attn_output, attn_output_weights = self.cross_attention(qry_reps, doc_reps, doc_reps)
        qry_reps = self.layerNorm1(qry_reps + attn_output)  # (qry_bsz, qry_len, vector_dim)

        if self.use_self_attention:
            attn_output, attn_output_weights = self.self_attention(qry_reps, qry_reps, qry_reps)
            qry_reps = self.layerNorm2(qry_reps + attn_output)  # (qry_bsz, qry_len, vector_dim)


        return qry_reps

    def final_scores(self, qry_reps):
        scores = self.linear(qry_reps[:, 0, :]).view(-1)  # (qry_bsz, 1) -> (qry_bsz)
        return scores

    def forward(self, data):
        # qry: desc

        qry_input, doc_input, labels = data

        qry_input = {key: value.to(self.device) for key, value in qry_input.items()}
        doc_input = {key: value.to(self.device) for key, value in doc_input.items() if key != 'filter_indices'}
        labels = labels.to(self.device)

        qry_reps = self.encoder(**qry_input, return_dict=True).last_hidden_state
        doc_reps = self.encoder(**doc_input, return_dict=True).last_hidden_state

        qry_reps = self.interaction_block(qry_reps, doc_reps)
        scores = self.final_scores(qry_reps)

        if self.loss_type == 'cross_entropy':
            loss = self.cross_entropy(scores, labels)
            n_correct = (torch.argmax(scores) == torch.tensor(0)).tolist()
            n_total = 1
        elif self.loss_type == 'bce':
            scores = self.sigmoid(scores)
            loss = self.bce_loss(scores, labels)
            n_correct = torch.sum((scores > 0.5) == (labels > 0.5)).tolist()
            n_total = len(scores)
        else:
            raise ValueError()

        return loss, n_correct, n_total

    def mask_sep(self, qry_attention_mask):
        sep_pos = qry_attention_mask.sum(1).unsqueeze(1) - 1  # the sep token position
        _zeros = torch.zeros_like(sep_pos)
        qry_attention_mask.scatter_(1, sep_pos.long(), _zeros)

        return qry_attention_mask

    def encode_code_synonyms(self, dataset, batch_size=200):
        """
        Encodes all code synonyms
        """
        c_desc_input_ids, c_desc_attention_mask = dataset.c_desc_input_ids, dataset.c_desc_attention_mask
        with torch.no_grad():
            # self.train(False)
            n_total = len(c_desc_input_ids)
            batch_size = (n_total - 1) // ((n_total - 1) // batch_size + 1) + 1

            doc_vectors = []
            for i in range(0, n_total, batch_size):
                desc_input_ids = c_desc_input_ids[i:i + batch_size].to(self.device)
                desc_attention_mask = c_desc_attention_mask[i:i + batch_size].to(self.device)

                if len(desc_input_ids) < batch_size:
                    n_fill = batch_size - len(desc_input_ids)
                    desc_input_ids = torch.cat([desc_input_ids] + [c_desc_input_ids[-1:].to(self.device)] * n_fill)
                    desc_attention_mask = torch.cat([desc_attention_mask] + [c_desc_attention_mask[-1:].to(self.device)] * n_fill)

                desc_inputs = {
                    'input_ids': desc_input_ids,
                    'attention_mask': desc_attention_mask
                }

                doc_vectors.append(self.encoder(**desc_inputs).last_hidden_state.to('cpu'))

            doc_vectors = torch.cat(doc_vectors, dim=0)[:n_total]
            return doc_vectors

    def calc_logits(self, note_inputs, c_descs_vectors, dataset, batch_size=100):
        # set model.train(False)
        with torch.no_grad():

            note_inputs = {key: value.to(self.device) for key, value in note_inputs.items() if key != 'filter_indices'}

            note_reps = self.encoder(**note_inputs, return_dict=True).last_hidden_state

            scores_list = []

            n_doc = len(c_descs_vectors)
            for i in range(0, n_doc, batch_size):
                c_descs_reps = c_descs_vectors[i:i + batch_size].to(self.device)

                if self.switch_qd:
                    reps = self.interaction_block(c_descs_reps, note_reps)
                else:
                    reps = self.interaction_block(note_reps, c_descs_reps)

                scores = self.final_scores(reps)
                scores_list.append(scores)

            scores_list = torch.cat(scores_list, dim=0)
        return scores_list


    def rank(self, dataset, c_descs_vectors, return_eval=True, save_topk=False):

        scores_list = []
        for i, index in enumerate(range(dataset.len)):
            note_inputs = dataset.get_note_inputs(index)
            scores = self.calc_logits(
                note_inputs,
                c_descs_vectors,
                dataset,
                batch_size=100)
            scores_list.append(scores)

            if i % 1000 == 0:
                print(i, end=' ')
        print()

        scores_list = torch.stack(scores_list, dim=0)

        if save_topk:
            dataset.ranks[:len(dataset)] = scores_list.argsort(descending=True, dim=1)[:, :dataset.n_ranks_ance].cpu().numpy()

        if return_eval:
            return self.eval(y=dataset.binary_labels[:dataset.len], yhat_raw=scores_list.to('cpu').numpy())

