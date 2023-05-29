import re
from typing import List, Tuple, Optional

from tqdm import tqdm


class TextUtils:
    @staticmethod
    def normalize_text(s: str) -> str:
        """Normalizes string, removes punctuation and
        non alphabet symbols

        Args:
            s (str): string to mormalize

        Returns:
            str: normalized string
        """
        s = s.lower()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Zа-яйёьъА-Яй]+", r" ", s)
        s = s.strip()
        return s


    @staticmethod
    def read_langs_pairs_from_file(filename: str):
        """Read lang from file

        Args:
            filename (str): path to dataset
            lang1 (str): name of first lang
            lang2 (str): name of second lang
            reverse (Optional[bool]): revers inputs (eng->ru of ru->eng)

        Returns:
            Tuple[Lang, Lang, List[Tuple[str, str]]]: tuple of
                (input lang class, out lang class, string pairs)
        """
        with open(filename, mode="r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        lang_pairs = []
        for line in tqdm(lines, desc="Reading from file"):
            lang_pair = tuple(map(TextUtils.normalize_text, line.split("\t")[:2]))
            lang_pairs.append(lang_pair)

        return lang_pairs

def short_text_filter_function(x, max_length, prefix_filter=None):
    len_filter = lambda x: len(x[0].split(" ")) <= max_length and len(x[1].split(" ")) <= max_length
    if prefix_filter:
        prefix_filter_func = lambda x: x[0].startswith(prefix_filter)
    else:
        prefix_filter_func = lambda x: True
    return len_filter(x) and prefix_filter_func(x)
