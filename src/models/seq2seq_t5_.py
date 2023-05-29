import torch

import metrics

from src.models.positional_encoding import PositionalEncoding


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        target_tokenizer,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = PositionalEncoding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = PositionalEncoding(max_len, embedding_size)
        self.max_len = max_len
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = 4
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_sched_step_every,
            gamma=lr_sched_gamma,
        )
        self.target_tokenizer = target_tokenizer
        
    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

    def training_step(self, batch):
        self.optimizer.zero_grad()
        input_tensor, target_tensor = batch
        (_, output) = self.forward(input_tensor, target_tensor[:, :-1])
        target = target_tensor[:, 1:].reshape(-1)
        output = output.reshape(-1, output.shape[-1])
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, batch):
        input_tensor, target_tensor = batch
        (_, output) = self.forward(input_tensor, target_tensor[:, :-1])
        target = target_tensor[:, 1:].reshape(-1)
        output = output.reshape(-1, output.shape[-1])
        loss = self.criterion(output, target)
        return loss.item()

    def translate(self, src):
        self.eval()

        with torch.no_grad():
            src = src.to(self.device)
            trg_input = torch.tensor(
                [[self.target_tokenizer.tokenizer.token_to_id("[BOS]")]],
                device=self.device,
            )
            output = []

            while (
                trg_input[:, -1].item()
                != self.target_tokenizer.tokenizer.token_to_id("[EOS]")
                and len(output) < self.max_len
            ):
                trg_input = trg_input.to(self.device)
                _, output_step = self.forward(src, trg_input)
                pred_token = torch.argmax(output_step, dim=-1)[:, -1]
                output.append(pred_token.item())

                trg_input = torch.cat((trg_input, pred_token.unsqueeze(1)), dim=-1)

            predicted_ids = torch.tensor(output, device=self.device).unsqueeze(0)

        self.train()
        return predicted_ids

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
