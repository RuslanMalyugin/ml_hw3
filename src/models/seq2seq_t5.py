import torch
from torch.optim.lr_scheduler import StepLR
from transformers import T5ForConditionalGeneration
from transformers.optimization import Adafactor

import src.metrics as metrics


class Seq2SeqT5(torch.nn.Module):
    def __init__(
        self,
        device,
        tokenizer,
        model_name="t5-small",
    ):
        super(Seq2SeqT5, self).__init__()

        # TODO: Реализуйте конструктор seq2seq t5
        self.device = device
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            device
        )
        self.tokenizer = tokenizer
        self.t5_model.resize_token_embeddings(len(self.tokenizer))
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(torch.randn(param.size()))

        self.optimizer = Adafactor(
            self.t5_model.parameters(),
            lr=0.0001,
            relative_step=False,
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=2000,
            gamma=0.1,
        )
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, input_data, target_data, attention_mask=None):
        output = self.t5_model(
            input_ids=input_data, attention_mask=attention_mask, labels=target_data
        )
        topi = torch.argmax(output.logits, dim=-1)
        return topi.clone(), output

    def training_step(self, batch):
        # TODO: Реализуйте оценку на 1 батче данных по примеру seq2seq_rnn.py
        self.t5_model.train()
        self.optimizer.zero_grad()

        src, trg, attention_mask = batch
        _, output = self(src, trg, attention_mask)

        loss = output.loss
        loss.backward()  # Backpropagate the gradients
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def validation_step(self, batch):
        # TODO: Реализуйте оценку на 1 батче данных по примеру seq2seq_rnn.py
        with torch.no_grad():
            (src, trg, attenion_mask) = batch
            _, output = self(src, trg, attenion_mask)
            loss = output.loss
        return loss.item()

    def generate(self, input_ids, attention_mask=None, max_length=50):
        return self.t5_model.generate(
            input_ids, attention_mask=attention_mask, max_length=max_length
        )

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = predicted.squeeze(-1).detach().cpu().numpy()
        actuals = target_tensor.squeeze(-1).detach().cpu().numpy()
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
