from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from tokenizers.processors import TemplateProcessing

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

class BPETokenizer:
    def __init__(self, sentence_list, max_sent_len):
        """
        sentence_list - список предложений для обучения
        """
        #components
        self.max_sent_len = max_sent_len
        self.spl_tokens = ["[UNK]", "[EOS]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.unk_token = "[UNK]"
        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        self.trainer = BpeTrainer(special_tokens=self.spl_tokens)
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.decoder = BPEDecoder(" ")
        self.tokenizer.train_from_iterator(sentence_list, trainer=self.trainer)
        self.word2index = self.tokenizer.get_vocab()
        self.index2word = {v: k for k, v in self.word2index.items()}

        
        # TODO: Реализуйте конструктор c помощью https://huggingface.co/docs/transformers/fast_tokenizers, обучите токенизатор, подготовьте нужные аттрибуты(word2index, index2word)

    def pad_sent(self, token_ids_list):
        if len(token_ids_list) < self.max_sent_len:
            padded_token_ids_list = token_ids_list + [self.word2index["[PAD]"]] * (self.max_sent_len - len(token_ids_list))
        else:
            padded_token_ids_list = token_ids_list[:self.max_sent_len - 1] + [self.word2index["[EOS]"]]
        return padded_token_ids_list
    
    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        tokenized_data = self.tokenizer.encode(sentence).ids
        return self.pad_sent(tokenized_data)
        # TODO: Реализуйте метод токенизации с помощью обученного токенизатора


    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        return self.tokenizer.decode(token_list).split()