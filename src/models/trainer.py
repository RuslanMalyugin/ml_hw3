from tqdm import tqdm



class Trainer:
    def __init__(self, model, model_config, logger):
        self.model = model
        self.epoch_num = model_config['epoch_num']
        self.logger = logger

        self.logger.log(model_config)

    def train(self, train_dataloader, val_dataloader):
        try:
            for epoch in tqdm(range(self.epoch_num)):
                train_epoch_loss = 0
                self.model.train()
                for batch in train_dataloader:
                    train_loss = self.model.training_step(batch)
                    train_epoch_loss += train_loss
                train_epoch_loss = train_epoch_loss / len(train_dataloader)

                val_epoch_loss, val_epoch_bleu = 0, 0
                self.model.eval()
                for batch in val_dataloader:
                    val_loss = self.model.validation_step(batch)
                    val_epoch_loss += val_loss
                val_epoch_loss = val_epoch_loss / len(val_dataloader)

                input_tensor, target_tensor = batch
                predicted_samples, _ = self.model.forward(input_tensor)
                bleu_score, actual_sentences, predicted_sentences = self.model.eval_bleu(predicted_samples, target_tensor)
                print('Current BLEU: ', bleu_score)
                for a, b in zip(actual_sentences[:5], predicted_sentences[:5]):
                    print(f"{a} ---> {b}")
                print('##############################')

                self.logger.log({"val_loss": val_epoch_loss,
                                 "train_loss": train_epoch_loss ,
                                 "bleu_score": bleu_score})

        except KeyboardInterrupt:
            pass

        print(f"Last {epoch} epoch train loss: ", train_epoch_loss)
        print(f"Last {epoch} epoch val loss: ", val_epoch_loss)
        print(f"Last {epoch} epoch val bleu: ", bleu_score)
