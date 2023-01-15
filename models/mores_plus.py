import torch
from evaluation import eval_func

from transformers.models.bart.modeling_bart import BartModel, BartClassificationHead


class MORES_PLUS(torch.nn.Module):
    def __init__(self, args, device, loss, switch_qd):
        super(MORES_PLUS, self).__init__()

        self.model = BartModel.from_pretrained(args.config_name)
        self.classification_head = BartClassificationHead(
            self.model.config.d_model, self.model.config.d_model, 1, 0.0
        )

        self.encoder = self.model.get_encoder()
        self.decoder = self.model.get_decoder()

        self.eval = eval_func

        self.loss_type = loss
        if loss == 'cross_entropy':
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        elif loss == 'bce':
            self.sigmoid = torch.nn.Sigmoid()
            self.bce_loss = torch.nn.BCELoss(reduction='mean')
        else:
            raise ValueError()

        self.device = device
        self.switch_qd = switch_qd
        self.vector_dim = args.vector_dim

    def decode(self, encoder_hidden_states, encoder_attention_mask, decoder_input_ids, decoder_attention_mask):
        hidden_states = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )[0]  # last hidden state

        eos_mask = decoder_input_ids.eq(self.model.config.eos_token_id).to(hidden_states.device)
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
        scores = self.classification_head(sentence_representation)
        return scores


    def forward(self, data):
        # qry: desc
        qry_inputs, doc_inputs, labels = data # medical note is the query, code labels are the documents
        labels = labels.to(self.device)

        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = \
            doc_inputs['input_ids'].to(self.device),\
            doc_inputs['attention_mask'].to(self.device),\
            qry_inputs['input_ids'].to(self.device), \
            qry_inputs['attention_mask'].to(self.device)

        decoder_bsz = decoder_input_ids.shape[0]
        if self.switch_qd:
            encoder_hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0]

            encoder_hidden_states = encoder_hidden_states.reshape((1, -1, self.vector_dim))  # (1, doc_len, vector_dim)
            encoder_attention_mask = attention_mask.reshape((1, -1))  # (1, doc_len)

            encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, decoder_bsz, dim=0)
            encoder_attention_mask = torch.repeat_interleave(encoder_attention_mask, decoder_bsz, dim=0)

            scores = self.decode(encoder_hidden_states, encoder_attention_mask, decoder_input_ids, decoder_attention_mask)
            scores = scores.view(-1)
        else:
            encoder_hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0]

            encoder_bsz = encoder_hidden_states.shape[0]

            scores_list = []
            for i in range(decoder_bsz):
                decoder_input_ids_i = decoder_input_ids[i].unsqueeze(0).repeat_interleave(encoder_bsz, dim=0)
                decoder_attention_mask_i = decoder_attention_mask[i].unsqueeze(0).repeat_interleave(encoder_bsz, dim=0)

                scores = self.decode(
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids_i,
                    decoder_attention_mask=decoder_attention_mask_i
                ).transpose(0, 1)
                scores_list.append(scores)
            scores = torch.cat(scores_list, dim=0) # (decoder_bsz, n_c_descs)

            scores = torch.max(scores, dim=0)[0] # (n_c_descs)

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

    def encode_code_synonyms(self, dataset, batch_size=200):
        if self.switch_qd:
            return None
        else:
            c_desc_input_ids, c_desc_attention_mask = dataset.c_desc_input_ids, dataset.c_desc_attention_mask
            with torch.no_grad():
                n_total = len(c_desc_input_ids)
                batch_size = (n_total - 1) // ((n_total - 1) // batch_size + 1) + 1

                encoder_hidden_states_list, encoder_attention_mask_list = [], []
                for i in range(0, n_total, batch_size):
                    desc_input_ids = c_desc_input_ids[i:i + batch_size].to(self.device)
                    desc_attention_mask = c_desc_attention_mask[i:i + batch_size].to(self.device)

                    if len(desc_input_ids) < batch_size:
                        n_fill = batch_size - len(desc_input_ids)
                        desc_input_ids = torch.cat([desc_input_ids] + [c_desc_input_ids[-1:].to(self.device)] * n_fill)
                        desc_attention_mask = torch.cat([desc_attention_mask] + [c_desc_attention_mask[-1:].to(self.device)] * n_fill)

                    encoder_hidden_states = self.encoder(
                        input_ids=desc_input_ids,
                        attention_mask=desc_attention_mask
                    )[0].to('cpu')

                    encoder_hidden_states_list.append(encoder_hidden_states)
                    encoder_attention_mask_list.append(desc_attention_mask)

                encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0)[:n_total]
                encoder_attention_mask = torch.cat(encoder_attention_mask_list, dim=0)[:n_total]
                return encoder_hidden_states, encoder_attention_mask

    def calc_logits(self, note_inputs, c_descs_vectors, dataset, batch_size=100):
        with torch.no_grad():
            n_c_descs = len(dataset.c_desc_input_ids)
            if self.switch_qd:
                input_ids, attention_mask = note_inputs['input_ids'].to(self.device), note_inputs['attention_mask'].to(self.device)

                decoder_bsz = batch_size
                encoder_hidden_states = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state

                encoder_hidden_states = encoder_hidden_states.reshape((1, -1, self.vector_dim))  # (1, doc_len, vector_dim)
                encoder_attention_mask = attention_mask.reshape((1, -1))

                encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, decoder_bsz, dim=0).to(self.device)
                encoder_attention_mask = torch.repeat_interleave(encoder_attention_mask, decoder_bsz, dim=0).to(self.device)

                scores_list = []
                for i in range(0, n_c_descs, batch_size):
                    size = min(batch_size, n_c_descs - i)

                    scores = self.decode(
                        encoder_hidden_states[:size],
                        encoder_attention_mask[:size],
                        dataset.c_desc_input_ids[i: i+batch_size].to(self.device),
                        dataset.c_desc_attention_mask[i: i+batch_size].to(self.device)
                    )
                    scores_list.append(scores.view(-1))
                scores_all = torch.cat(scores_list, dim=0)[:n_c_descs]
            else:
                encoder_hidden_states, encoder_attention_mask = c_descs_vectors
                decoder_input_ids, decoder_attention_mask = note_inputs['input_ids'].to(self.device), note_inputs['attention_mask'].to(self.device)

                scores_list_list = []
                decoder_bsz = decoder_input_ids.shape[0]
                for i in range(decoder_bsz):
                    decoder_input_ids_i = decoder_input_ids[i].unsqueeze(0).repeat_interleave(batch_size, dim=0).to(self.device)
                    decoder_attention_mask_i = decoder_attention_mask[i].unsqueeze(0).repeat_interleave(batch_size, dim=0).to(self.device)

                    scores_list = []
                    for j in range(0, n_c_descs, batch_size):
                        size = min(batch_size, n_c_descs - i)

                        scores = self.decode(
                            encoder_hidden_states=encoder_hidden_states[:batch_size].to(self.device),
                            encoder_attention_mask=encoder_attention_mask[:batch_size].to(self.device),
                            decoder_input_ids=decoder_input_ids_i[:size],
                            decoder_attention_mask=decoder_attention_mask_i[:size]
                        ).transpose(0, 1)
                        scores_list.append(scores)
                    scores_list = torch.cat(scores_list, dim=1)[:,:n_c_descs]

                    scores_list_list.append(scores_list)
                scores_all = torch.cat(scores_list_list, dim=0)
                scores_all = torch.max(scores_all, dim=0)[0]

        return scores_all

    def rank(self, dataset, c_descs_vectors, return_eval=True, save_topk=False):

        scores_list = []
        for i, index in enumerate(range(len(dataset))):
            note_inputs = dataset.get_note_inputs(index)
            scores = self.calc_logits(
                note_inputs,
                c_descs_vectors,
                dataset,
                batch_size=10
            )
            scores_list.append(scores)

            print(i, end=' ')
        print()
        scores_list = torch.stack(scores_list, dim=0)


        if save_topk:
            dataset.ranks[:len(dataset)] = scores_list.argsort(descending=True, dim=1)[:, :dataset.n_ranks_ance].cpu().numpy()

        if return_eval:
            return self.eval(y=dataset.binary_labels[:len(dataset)], yhat_raw=scores_list.to('cpu').numpy())
