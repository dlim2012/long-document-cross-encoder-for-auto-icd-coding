
import torch
from torch import Tensor
from torch.cuda.amp import autocast
from evaluation import eval_func

from transformers import AutoConfig, AutoModel, AutoTokenizer

class COIL(torch.nn.Module):
    def __init__(self, args, device, loss, switch_qd, chunk_notes):
        super(COIL, self).__init__()

        config = AutoConfig.from_pretrained(args.config_name)
        self.encoder = AutoModel.from_pretrained(args.config_name, config=config)

        self.tokenizer = AutoTokenizer.from_pretrained(args.config_name)

        assert loss in ['cross_entropy', 'bce']
        if loss == 'cross_entropy':
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            self.sigmoid = torch.nn.Sigmoid()
            self.bce_loss = torch.nn.BCELoss(reduction='mean')

        self.tok_proj = torch.nn.Linear(args.vector_dim, args.coil_token_dim)
        self.cls_proj = torch.nn.Linear(args.vector_dim, args.coil_cls_dim)

        self.eval = eval_func

        self.args = args
        self.device = device
        self.loss_type = loss
        self.switch_qd = switch_qd
        self.chunk_notes = chunk_notes

    def vector_representation(self, input):
        out = self.encoder(**input, return_dict=True).last_hidden_state
        cls = self.cls_proj(out[:, 0])
        reps = self.tok_proj(out)  # D * LD * d
        return cls, reps

    def forward(self, data):
        # qry: desc
        n_correct, n_total = 0, 0

        qry_input, doc_input, labels = data # without switch_qd: medical note is the query, code labels are the documents
        labels = labels.to(self.device)
        qry_input = {key: value.to(self.device) for key, value in qry_input.items()}
        doc_input = {key: value.to(self.device) for key, value in doc_input.items() if key != 'filter_indices'}

        qry_cls, qry_reps = self.vector_representation(qry_input)
        doc_cls, doc_reps = self.vector_representation(doc_input)

        # mask ingredients
        doc_input_ids: Tensor = doc_input['input_ids'].to(self.device)
        qry_input_ids: Tensor = qry_input['input_ids'].to(self.device)
        qry_attention_mask: Tensor = qry_input['attention_mask'].to(self.device)

        self.mask_sep(qry_attention_mask)

        tok_scores = self.compute_tok_score_cart(
            doc_reps, doc_input_ids,
            qry_reps, qry_input_ids, qry_attention_mask
        )
        cls_scores = torch.mm(qry_cls, doc_cls.T)

        with autocast(False):
            scores = tok_scores.float() + cls_scores.float()  # Q * D

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
        elif self.loss_type == 'bce':
            scores = self.sigmoid(scores)
            loss = self.bce_loss(scores, labels)
            n_correct += torch.sum((scores > 0.5) == (labels > 0.5)).tolist()
            n_total += len(scores)
        else:
            raise ValueError()

        return loss, n_correct, n_total


    def mask_sep(self, qry_attention_mask):
        sep_pos = qry_attention_mask.sum(1).unsqueeze(1) - 1  # the sep token position
        _zeros = torch.zeros_like(sep_pos)
        qry_attention_mask.scatter_(1, sep_pos.long(), _zeros)

        return qry_attention_mask

    def compute_tok_score_pair(self, doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask):
        exact_match = qry_input_ids.unsqueeze(2) == doc_input_ids.unsqueeze(1)  # B * LQ * LD
        exact_match = exact_match.float()
        # qry_reps: B * LQ * d
        # doc_reps: B * LD * d
        scores_no_masking = torch.bmm(qry_reps, doc_reps.permute(0, 2, 1))  # B * LQ * LD
        tok_scores, _ = (scores_no_masking * exact_match).max(dim=2)  # B * LQ
        # remove padding and cls token
        tok_scores = (tok_scores * qry_attention_mask)[:, 1:].sum(-1)
        return tok_scores

    def compute_tok_score_cart(self, doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask):
        qry_input_ids = qry_input_ids.unsqueeze(2).unsqueeze(3)  # Q * LQ * 1 * 1
        doc_input_ids = doc_input_ids.unsqueeze(0).unsqueeze(1)  # 1 * 1 * D * LD
        exact_match = doc_input_ids == qry_input_ids  # Q * LQ * D * LD
        exact_match = exact_match.float()
        scores_no_masking = torch.matmul(
            qry_reps.view(-1, self.args.coil_token_dim),  # (Q * LQ) * d
            doc_reps.view(-1, self.args.coil_token_dim).transpose(0, 1)  # d * (D * LD)
        )
        scores_no_masking = scores_no_masking.view(
            *qry_reps.shape[:2], *doc_reps.shape[:2])  # Q * LQ * D * LD
        # scores_no_masking = scores_no_masking.permute(0, 2, 1, 3)  # Q * D * LQ * LD
        scores, _ = (scores_no_masking * exact_match).max(dim=3)  # Q * LQ * D
        tok_scores = (scores * qry_attention_mask.unsqueeze(2))[:, 1:].sum(1)
        return tok_scores

    def encode_code_synonyms(self, dataset, batch_size=200):
        """
        code descriptions are the documents
        """
        c_desc_input_ids, c_desc_attention_mask = dataset.c_desc_input_ids, dataset.c_desc_attention_mask
        with torch.no_grad():
            # self.train(False)
            n_total = len(c_desc_input_ids)
            batch_size = (n_total - 1) // ((n_total - 1) // batch_size + 1) + 1

            cls_vectors, tok_vectors = [], []
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
                cls, reps = self.vector_representation(desc_inputs)
                cls_vectors.append(cls.to('cpu'))
                tok_vectors.append(reps.to('cpu'))

            cls_vectors = torch.cat(cls_vectors, dim=0)[:n_total]
            tok_vectors = torch.cat(tok_vectors, dim=0)[:n_total]

        return (cls_vectors, tok_vectors)

    def calc_logits(self, note_inputs, c_descs_vectors, dataset, batch_size=100):

        doc_cls_vectors, doc_reps_vectors = c_descs_vectors
        with torch.no_grad():
            note_inputs = {key: value.to(self.device) for key, value in note_inputs.items() if key != 'filter_indices'}

            # encode qry
            qry_cls, qry_reps = self.vector_representation(note_inputs)

            qry_input_ids: Tensor = note_inputs['input_ids'].to(self.device)

            if not self.switch_qd:
                qry_attention_mask = note_inputs['attention_mask'].to(self.device)
                self.mask_sep(qry_attention_mask)

            scores_list = []

            for i in range(0, len(doc_cls_vectors), batch_size):
                doc_cls = doc_cls_vectors[i: i+batch_size].to(self.device)
                doc_reps = doc_reps_vectors[i: i+batch_size].to(self.device)

                doc_input_ids: Tensor = dataset.c_desc_input_ids[i:i+batch_size].to(self.device)

                if self.switch_qd:
                    doc_attention_mask: Tensor = dataset.c_desc_attention_mask[i:i + batch_size].to(self.device)
                    self.mask_sep(doc_attention_mask)

                    tok_scores = self.compute_tok_score_cart(
                        qry_reps, qry_input_ids,
                        doc_reps, doc_input_ids, doc_attention_mask
                    )
                    cls_scores = torch.mm(doc_cls, qry_cls.T)
                else:
                    tok_scores = self.compute_tok_score_cart(
                        doc_reps, doc_input_ids,
                        qry_reps, qry_input_ids, qry_attention_mask
                    )
                    cls_scores = torch.mm(qry_cls, doc_cls.T)

                with autocast(False):
                    scores = tok_scores + cls_scores  # B

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

