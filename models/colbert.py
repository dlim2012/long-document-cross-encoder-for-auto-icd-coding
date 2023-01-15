import torch
from torch import Tensor
from evaluation import eval_func

from transformers import AutoConfig, AutoModel, AutoTokenizer

class ColBERT(torch.nn.Module):
    def __init__(self, args, device, chunk_notes):
        super(ColBERT, self).__init__()

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

        self.use_cls = (args.colbert_cls_dim > 0)
        if args.colbert_cls_dim > 0:
            self.cls_proj = torch.nn.Linear(args.vector_dim, args.colbert_cls_dim)
        self.tok_proj = torch.nn.Linear(args.vector_dim, args.colbert_token_dim)

        self.ln_tok = torch.nn.LayerNorm(args.colbert_token_dim)

        self.eval = eval_func

        self.args = args
        self.device = device
        self.loss_type = args.loss
        self.switch_qd = args.switch_qd
        self.sample_method = args.sample_method
        self.chunk_notes = chunk_notes

    def vector_representation(self, input):
        out = self.encoder(**input, return_dict=True)
        out_reps = self.tok_proj(out.last_hidden_state)
        out_reps = torch.nn.functional.normalize(out_reps, p=2, dim=2)
        if self.use_cls:
            out_cls = self.cls_proj(out.last_hidden_state[:, 0])
            return out_cls, out_reps
        return out_reps

    def forward(self, data):
        # qry: desc
        n_correct, n_total = 0, 0

        qry_input, doc_input, labels = data
        # if switched: code labels are the queries, medical notes are the documents
        # if not switched: medical notes are the queries, code labels are the documents
        if self.switch_qd and self.args.use_filter: # medical note is doc
            filter_indices = doc_input['filter_indices']
            filter_indices = filter_indices.to(self.device).unsqueeze(len(filter_indices.shape))

        qry_input = {key: value.to(self.device) for key, value in qry_input.items()}
        doc_input = {key: value.to(self.device) for key, value in doc_input.items() if key != 'filter_indices'}
        labels = labels.to(self.device)

        if self.use_cls:
            qry_cls, qry_reps = self.vector_representation(qry_input)
            doc_cls, doc_reps = self.vector_representation(doc_input)
        else:
            qry_reps = self.vector_representation(qry_input)
            doc_reps = self.vector_representation(doc_input)

        if self.switch_qd and self.args.use_filter:  # medical note is doc
            doc_reps = doc_reps * filter_indices

        qry_attention_mask: Tensor = qry_input['attention_mask'].to(self.device)
        self.mask_sep(qry_attention_mask)

        scores = self.compute_tok_score_cart(doc_reps, qry_reps, qry_attention_mask)

        if self.sample_method == 'in-batch':
            if self.use_cls:
                raise NotImplementedError()

            if self.chunk_notes:
                raise NotImplementedError()

            scores = scores.to('cpu')
            x = torch.sum(torch.softmax(scores, dim=1) * torch.eye(scores.shape[0]), dim=1)
            x = -1 * torch.log(x)
            loss = torch.mean(x)

            n_correct = torch.sum(torch.argmax(scores, dim=1) == torch.arange(scores.shape[0])).tolist()
            n_total = scores.shape[0]

        else:
            if self.use_cls:
                scores += torch.mm(qry_cls, doc_cls.T)

            if self.chunk_notes:
                if self.switch_qd:
                    scores = scores.max(axis=1)[0]
                else:
                    scores = scores.max(axis=0)[0]
            scores = scores.view(-1)

            if self.loss_type == 'cross_entropy':
                loss = self.cross_entropy(scores, labels)
                n_correct += (torch.argmax(scores) == torch.tensor(0)).tolist()
                n_total += 1
            else:
                scores = self.sigmoid(scores/100)
                loss = self.bce_loss(scores, labels)
                n_correct += torch.sum((scores > 0.5) == (labels > 0.5)).tolist()
                n_total += len(scores)

        return loss, n_correct, n_total

    def mask_sep(self, qry_attention_mask):
        sep_pos = qry_attention_mask.sum(1).unsqueeze(1) - 1  # the sep token position
        _zeros = torch.zeros_like(sep_pos)
        qry_attention_mask.scatter_(1, sep_pos.long(), _zeros)

        return qry_attention_mask

    def compute_tok_score_cart(self, doc_reps, qry_reps, qry_attention_mask):
        scores_no_masking = torch.matmul(
            qry_reps.view(-1, self.args.colbert_token_dim),  # (Q * LQ) * d
            doc_reps.view(-1, self.args.colbert_token_dim).transpose(0, 1)  # d * (D * LD)
        )
        scores_no_masking = scores_no_masking.view(
            *qry_reps.shape[:2], *doc_reps.shape[:2])  # Q * LQ * D * LD
        scores, _ = scores_no_masking.max(dim=3)  # Q * LQ * D

        # remove scores for cls token and pad tokens
        tok_scores = (scores * qry_attention_mask.unsqueeze(2))[:, 1:].sum(1)
        return tok_scores

    def encode_code_synonyms(self, dataset, batch_size=200):
        """
        Encodes all code synonyms
        """
        c_desc_input_ids, c_desc_attention_mask = dataset.c_desc_input_ids, dataset.c_desc_attention_mask
        with torch.no_grad():
            # self.train(False)
            n_total = len(c_desc_input_ids)
            batch_size = (n_total - 1) // ((n_total - 1) // batch_size + 1) + 1


            if self.use_cls:
                doc_cls_vectors, doc_reps_vectors = [], []
            else:
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

                if self.use_cls:
                    doc_cls, doc_reps = self.vector_representation(desc_inputs)
                    doc_cls_vectors.append(doc_cls.to('cpu'))
                    doc_reps_vectors.append(doc_reps.to('cpu'))
                else:
                    doc_vectors.append(self.vector_representation(desc_inputs).to('cpu'))

            if self.use_cls:
                doc_cls_vectors = torch.cat(doc_cls_vectors, dim=0)[:n_total]
                doc_reps_vectors = torch.cat(doc_reps_vectors, dim=0)[:n_total]
                return doc_cls_vectors, doc_reps_vectors
            else:
                doc_vectors = torch.cat(doc_vectors, dim=0)[:n_total]
                return doc_vectors

    def calc_logits(self, note_inputs, c_descs_vectors, dataset, batch_size=100):
        """
        qry_input is a discharge note
        """
        # set model.train(False)
        with torch.no_grad():
            if self.switch_qd and self.args.use_filter:
                filter_indices = note_inputs['filter_indices']
                filter_indices = filter_indices.to(self.device).unsqueeze(len(filter_indices.shape))

            note_inputs = {key: value.to(self.device) for key, value in note_inputs.items() if key != 'filter_indices'}

            if self.use_cls:
                qry_cls, qry_reps = self.vector_representation(note_inputs)
            else:
                qry_reps = self.vector_representation(note_inputs)

            qry_attention_mask = note_inputs['attention_mask']
            self.mask_sep(qry_attention_mask)

            if self.switch_qd and self.args.use_filter: # only when medical notes are used as documents
                qry_reps = qry_reps * filter_indices

            scores_list = []

            if self.use_cls:
                doc_cls_vectors, doc_reps_vectors = c_descs_vectors
                n_doc = len(doc_cls_vectors)
            else:
                n_doc = len(c_descs_vectors)
            for i in range(0, n_doc, batch_size):
                if self.use_cls:
                    doc_cls = doc_cls_vectors[i: i + batch_size].to(self.device)
                    doc_reps = doc_reps_vectors[i:i + batch_size].to(self.device)
                else:
                    doc_reps = c_descs_vectors[i:i + batch_size].to(self.device)

                if self.switch_qd:
                    doc_attention_mask = dataset.c_desc_attention_mask[i:i+batch_size].to(self.device)
                    self.mask_sep(doc_attention_mask)
                    scores = self.compute_tok_score_cart(qry_reps, doc_reps, doc_attention_mask)
                else:
                    scores = self.compute_tok_score_cart(doc_reps, qry_reps, qry_attention_mask)

                if self.use_cls:
                    scores += torch.mm(qry_cls, doc_cls.T)

                if self.chunk_notes:
                    if self.switch_qd:
                        scores = scores.max(axis=1)[0]
                    else:
                        scores = scores.max(axis=0)[0]

                scores_list.append(scores.view(-1))

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

