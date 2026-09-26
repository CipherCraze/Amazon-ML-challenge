import re
import math
from collections import Counter
import unicodedata

# Regex patterns
PIN_INDIA_REGEX = re.compile(r'\b[1-9][0-9]{5}\b')
ZIP_US_REGEX = re.compile(r'\b\d{5}(?:-\d{4})?\b')
NON_LATIN_REGEX = re.compile(r'[\u0900-\u0D7F\u0E00-\u0E7F\u4E00-\u9FFF\u0600-\u06FF\u0400-\u04FF]')

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'enterprises', 'holdings', 'group',
    'services', 'solutions', 'technologies', 'industries', 'partners',
    'associates', 'llp', 'pllc', 'gmbh', 'sa', 'bv', 'nv', 'spa', 'srl'
}

def clean_token(token: str) -> str:
    return re.sub(r'[^\w]', '', token.lower())

def extract_tokens(text: str) -> list[str]:
    if not text:
        return []
    toks = re.findall(r'\b\w+\b', text.lower())
    return [t for t in toks if t and t not in LEGAL_SUFFIXES]

def extract_all_tokens(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())

def extract_char_ngrams(text: str, n: int = 3) -> set[str]:
    clean = re.sub(r'[^a-z0-9]', '', text.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def extract_postal_code(addr: str, country: str) -> str | None:
    if not addr:
        return None
    if country == 'India':
        m = PIN_INDIA_REGEX.search(addr)
        return m.group(0) if m else None
    elif country == 'US':
        m = ZIP_US_REGEX.search(addr)
        if m:
            return m.group(0).split('-')[0]
    return None

def extract_region(addr: str, country: str) -> str | None:
    if not addr:
        return None
    toks = extract_all_tokens(addr)
    # Check for known state/region abbreviations or names
    us_states = {'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy', 'tx', 'ca', 'ny', 'fl', 'il'}
    in_states = {'maharashtra', 'delhi', 'karnataka', 'tamil nadu', 'telangana', 'gujarat', 'uttar pradesh', 'west bengal', 'haryana', 'kerala', 'rajasthan', 'andhra pradesh', 'punjab', 'odisha', 'bihar', 'assam', 'jharkhand', 'chhattisgarh', 'uttarakhand', 'goa', 'mh', 'dl', 'ka', 'tn', 'ts', 'tg', 'gj', 'up', 'wb', 'hr', 'kl', 'rj', 'ap', 'pb'}
    addr_lower = addr.lower()
    if country == 'US':
        for t in reversed(toks):
            if t in us_states:
                return t
    elif country == 'India':
        for st in in_states:
            if st in addr_lower:
                return st
    return None

def jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    k = 0
    transpositions = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2.0) / matches) / 3.0

    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * prefix_weight * (1.0 - jaro)

def levenshtein_ratio(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    dp = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        prev = dp[0]
        dp[0] = i
        c1 = s1[i - 1]
        for j in range(1, len2 + 1):
            temp = dp[j]
            cost = 0 if c1 == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp

    dist = dp[len2]
    return 1.0 - (dist / max(len1, len2))

def token_jaccard(toks1: list[str], toks2: list[str]) -> float:
    set1 = set(toks1)
    set2 = set(toks2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    ng1 = extract_char_ngrams(s1, n)
    ng2 = extract_char_ngrams(s2, n)
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)

def tfidf_cosine(toks1: list[str], toks2: list[str], idf_map: dict[str, float] = None) -> float:
    if not toks1 or not toks2:
        return 0.0
    c1 = Counter(toks1)
    c2 = Counter(toks2)
    
    shared = set(c1.keys()) & set(c2.keys())
    if not shared:
        return 0.0
    
    dot = 0.0
    for t in shared:
        w = idf_map.get(t, 1.0) if idf_map else 1.0
        dot += (c1[t] * w) * (c2[t] * w)
        
    norm1 = sum((count * (idf_map.get(t, 1.0) if idf_map else 1.0)) ** 2 for t, count in c1.items())
    norm2 = sum((count * (idf_map.get(t, 1.0) if idf_map else 1.0)) ** 2 for t, count in c2.items())
    
    denom = math.sqrt(norm1) * math.sqrt(norm2)
    return (dot / denom) if denom > 0 else 0.0

def detect_script(text: str) -> str:
    if not text:
        return "Empty"
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return "Devanagari"
        elif 0x0C00 <= code <= 0x0C7F:
            return "Telugu"
        elif 0x0B80 <= code <= 0x0BFF:
            return "Tamil"
        elif 0x0C80 <= code <= 0x0CFF:
            return "Kannada"
        elif 0x0D00 <= code <= 0x0D7F:
            return "Malayalam"
        elif 0x0A80 <= code <= 0x0AFF:
            return "Gujarati"
        elif 0x0980 <= code <= 0x09FF:
            return "Bengali"
        elif 0x0A00 <= code <= 0x0A7F:
            return "Gurmukhi"
        elif 0x0B00 <= code <= 0x0B7F:
            return "Oriya"
    return "Latin"
