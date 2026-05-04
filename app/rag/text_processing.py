import re

import pymorphy3


morph = pymorphy3.MorphAnalyzer()


def normalize_text(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    words = re.findall(r"[а-яa-z0-9]+", text)

    normalized_words = []
    for word in words:
        if re.fullmatch(r"[а-я]+", word):
            normalized_words.append(morph.parse(word)[0].normal_form)
        else:
            normalized_words.append(word)

    return normalized_words


def text_has_label(text: str, label: str) -> bool:
    text_tokens = set(normalize_text(text))
    label_tokens = set(normalize_text(label))
    return bool(text_tokens & label_tokens)
