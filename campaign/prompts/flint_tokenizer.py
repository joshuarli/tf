"""Byte-level BPE read from a checkpoint's tokenizer.json, for sizing corpus text before the served tokenizer confirms it."""

import json
import unicodedata
from pathlib import Path

CJK_RANGES = ((0x4E00, 0x9FA5), (0x3040, 0x309F), (0x30A0, 0x30FF))
SUFFIXES = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def bytes_to_unicode() -> dict[int, str]:
    listed = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = listed[:]
    extra = 0
    for byte in range(256):
        if byte not in listed:
            listed.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(listed, (chr(c) for c in chars)))


def category(ch: str) -> str:
    return unicodedata.category(ch)


def is_letter(ch: str) -> bool:
    return category(ch).startswith("L")


def is_number(ch: str) -> bool:
    return category(ch).startswith("N")


def is_upper_class(ch: str) -> bool:
    c = category(ch)
    return c in ("Lu", "Lt", "Lm", "Lo") or c.startswith("M")


def is_lower_class(ch: str) -> bool:
    c = category(ch)
    return c in ("Ll", "Lm", "Lo") or c.startswith("M")


def is_punct_sym(ch: str) -> bool:
    c = category(ch)
    return c.startswith("P") or c.startswith("S")


def is_cjk(ch: str) -> bool:
    return any(start <= ord(ch) <= end for start, end in CJK_RANGES)


class FlintTokenizer:
    """The checkpoint's Qwen-style pre-tokenizer and merge ranks; no normalization step in this tokenizer."""

    def __init__(self, tokenizer_json: Path):
        document = json.loads(Path(tokenizer_json).read_bytes())
        model = document["model"]
        self.vocab = model["vocab"]
        self.vocab_tokens = set(self.vocab)
        self.ranks = {(left, right): rank for rank, (left, right) in enumerate(model["merges"])}
        self.added = [token["content"] for token in document.get("added_tokens") or []]
        self.byte_map = bytes_to_unicode()
        self.reverse_map = {char: byte for byte, char in self.byte_map.items()}
        self.id_to_token = {index: token for token, index in self.vocab.items()}
        self.id_to_token.update({token["id"]: token["content"] for token in document.get("added_tokens") or []})
        self.token_to_id = {token: index for index, token in self.id_to_token.items()}

    def _numbers(self, text: str) -> list[str]:
        out, i = [], 0
        while i < len(text):
            if not is_number(text[i]):
                start = i
                while i < len(text) and not is_number(text[i]):
                    i += 1
                out.append(text[start:i])
                continue
            start = i
            while i < len(text) and is_number(text[i]):
                i += 1
            for at in range(start, i, 3):
                out.append(text[at:min(at + 3, i)])
        return out

    def _cjk(self, text: str) -> list[str]:
        out, i = [], 0
        while i < len(text):
            start = i
            inside = is_cjk(text[i])
            while i < len(text) and is_cjk(text[i]) == inside:
                i += 1
            out.append(text[start:i])
        return out

    def _match(self, text: str, at: int) -> int | None:
        n = len(text)
        head = text[at]
        first = head not in "\r\n" and not is_letter(head) and not is_number(head)

        def suffix(pos: int) -> int:
            for candidate in SUFFIXES:
                if text[pos:pos + len(candidate)].lower() == candidate:
                    return pos + len(candidate)
            return pos

        for consume in ((True, False) if first else (False,)):
            base = at + 1 if consume else at
            upper = base
            while upper < n and is_upper_class(text[upper]):
                upper += 1
            for cut in range(upper, base - 1, -1):
                if cut < n and is_lower_class(text[cut]):
                    end = cut
                    while end < n and is_lower_class(text[end]):
                        end += 1
                    return suffix(end)
        for consume in ((True, False) if first else (False,)):
            base = at + 1 if consume else at
            upper = base
            while upper < n and is_upper_class(text[upper]):
                upper += 1
            if upper > base:
                end = upper
                while end < n and is_lower_class(text[end]):
                    end += 1
                return suffix(end)
        for consume in ((True, False) if head == " " else (False,)):
            base = at + 1 if consume else at
            if base < n and is_punct_sym(text[base]):
                end = base
                while end < n and is_punct_sym(text[end]):
                    end += 1
                while end < n and text[end] in "\r\n":
                    end += 1
                return end
        if head.isspace():
            end = at
            while end < n and text[end].isspace():
                end += 1
            for cut in range(end - 1, at - 1, -1):
                if text[cut] in "\r\n":
                    stop = cut
                    while stop < end and text[stop] in "\r\n":
                        stop += 1
                    return stop
            if end == n:
                return end
            if end - 1 >= at + 1:
                return end - 1
            return end
        return None

    def _split(self, text: str) -> list[str]:
        out, i, gap = [], 0, 0
        while i < len(text):
            end = self._match(text, i)
            if end is None:
                i += 1
                continue
            if gap < i:
                out.append(text[gap:i])
            out.append(text[i:end])
            i = gap = end
        if gap < len(text):
            out.append(text[gap:])
        return out

    def _merge(self, symbols: list[str]) -> list[str]:
        while len(symbols) > 1:
            best, best_at = None, None
            for at in range(len(symbols) - 1):
                rank = self.ranks.get((symbols[at], symbols[at + 1]))
                if rank is not None and (best is None or rank < best):
                    best, best_at = rank, at
            if best is None:
                break
            left, right = symbols[best_at], symbols[best_at + 1]
            merged, at = [], 0
            while at < len(symbols):
                if at + 1 < len(symbols) and symbols[at] == left and symbols[at + 1] == right:
                    merged.append(left + right)
                    at += 2
                else:
                    merged.append(symbols[at])
                    at += 1
            symbols = merged
        return symbols

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self._numbers(text):
            for run in self._cjk(piece):
                for chunk in self._split(run):
                    symbols = [self.byte_map[byte] for byte in chunk.encode("utf-8")]
                    for token in self._merge(symbols):
                        ids.append(self.token_to_id[token])
        return ids

    def token_bytes(self, index: int) -> bytes:
        token = self.id_to_token[index]
        if token in self.vocab_tokens:
            return bytes(self.reverse_map[char] for char in token)
        return token.encode("utf-8")

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.token_bytes(index) for index in ids).decode("utf-8", errors="strict")

    def prefix_text(self, text: str, tokens: int) -> tuple[str, int]:
        """The longest text prefix that decodes cleanly from exactly ``tokens`` tokens of ``text``."""
        ids = self.encode(text)
        keep = min(tokens, len(ids))
        while keep > 0:
            try:
                return self.decode(ids[:keep]), keep
            except UnicodeDecodeError:
                keep -= 1
        return "", 0
