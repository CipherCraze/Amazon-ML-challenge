import re
import unicodedata
from collections import Counter
import array

# Regex patterns
PIN_INDIA_REGEX = re.compile(r'\b[1-9][0-9]{5}\b')
ZIP_US_REGEX = re.compile(r'\b\d{5}(?:-\d{4})?\b')

# Address generic words that appear in thousands of addresses
ADDRESS_GENERIC_WORDS = {
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'boulevard', 'blvd',
    'drive', 'dr', 'lane', 'ln', 'court', 'ct', 'place', 'pl', 'way',
    'highway', 'hwy', 'expressway', 'route', 'rt', 'parkway', 'pkwy',
    'floor', 'fl', 'flr', 'ground', 'first', 'second', 'third', 'fourth',
    'building', 'bldg', 'tower', 'complex', 'plaza', 'center', 'centre',
    'near', 'opp', 'opposite', 'behind', 'beside', 'next', 'adjacent',
    'plot', 'flat', 'shop', 'suite', 'ste', 'apt', 'apartment', 'unit',
    'no', 'number', 'h', 'house', 'room', 'block', 'sector', 'sec',
    'cross', 'main', 'colony', 'nagar', 'layout', 'enclave', 'vihar',
    'dist', 'district', 'state', 'india', 'us', 'usa', 'united', 'states',
    'city', 'town', 'village', 'post', 'po', 'pin', 'zip', 'code'
}

class AddressNormalizer:
    def __init__(self):
        pass

    def normalize_unicode(self, text: str) -> str:
        if not text:
            return ""
        decomposed = unicodedata.normalize('NFKD', text)
        return "".join(c for c in decomposed if not unicodedata.combining(c))

    def clean_address(self, raw_addr: str) -> str:
        if not raw_addr:
            return ""
        text = self.normalize_unicode(raw_addr).lower().strip()
        # Replace punctuation with spaces
        text = re.sub(r'[\-_/\\,;:.\'"&|()\[\]{}+*#@!~?<>^%$`=]', ' ', text)
        return " ".join(text.split())

    def extract_postal_code(self, raw_addr: str, country: str) -> str | None:
        if not raw_addr:
            return None
        if country == "India":
            m = PIN_INDIA_REGEX.search(raw_addr)
            return m.group(0) if m else None
        elif country == "US":
            m = ZIP_US_REGEX.search(raw_addr)
            return m.group(0).split('-')[0] if m else None
        return None

    def extract_tokens(self, clean_addr: str, min_len: int = 2, filter_generic: bool = True) -> list[str]:
        if not clean_addr:
            return []
        toks = clean_addr.split()
        res = []
        for t in toks:
            if len(t) < min_len:
                continue
            if filter_generic and t in ADDRESS_GENERIC_WORDS:
                continue
            res.append(t)
        return res

    def extract_token_pairs(self, clean_addr: str, max_pairs: int = 15, filter_generic: bool = True) -> list[tuple[str, str]]:
        """Extract ordered pairs of distinctive address tokens."""
        toks = self.extract_tokens(clean_addr, min_len=2, filter_generic=filter_generic)
        # Deduplicate while preserving order
        seen = set()
        unique_toks = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                unique_toks.append(t)
        
        pairs = []
        # Generate pairs (sorted lexically to be symmetric)
        n = len(unique_toks)
        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = sorted([unique_toks[i], unique_toks[j]])
                pairs.append((t1, t2))
                if len(pairs) >= max_pairs:
                    return pairs
        return pairs
