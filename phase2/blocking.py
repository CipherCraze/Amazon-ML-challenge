import os
import sys
import re
import unicodedata
import array
from collections import defaultdict, Counter
import ctypes
from ctypes import wintypes

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Windows memory tracking via native ctypes GetProcessMemoryInfo
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]

_psapi = ctypes.windll.psapi
_kernel32 = ctypes.windll.kernel32
_GetProcessMemoryInfo = _psapi.GetProcessMemoryInfo
_GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
_GetProcessMemoryInfo.restype = wintypes.BOOL

def get_memory_info_mb():
    """Return (current_rss_mb, peak_rss_mb) on Windows using GetProcessMemoryInfo."""
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    if _GetProcessMemoryInfo(_kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return (round(pmc.WorkingSetSize / (1024 * 1024), 2),
                round(pmc.PeakWorkingSetSize / (1024 * 1024), 2))
    return (0.0, 0.0)


# =====================================================================
# DETERMINISTIC NORMALIZER
# =====================================================================

LEGAL_SUFFIX_SET = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "pvt", "private", "co", "company", "llp", "gmbh", "sa", "sarl", "enterprises",
    "services", "solutions", "holdings", "group"
}

# Strict legal suffix tokens that specifically denote corporate structures
STRICT_LEGAL_TERMS = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "pvt", "private", "co", "company", "llp", "gmbh", "sa", "sarl"
}

URL_REGEX = re.compile(r'https?://[^\s]+|www\.[^\s]+', re.IGNORECASE)
DOMAIN_SUFFIX_REGEX = re.compile(r'\.(com|in|org|net|fr|co|io|biz|info|us)\b', re.IGNORECASE)
BRACKETED_PREFIX_REGEX = re.compile(r'^\s*\[(.*?)\]\s*', re.IGNORECASE)

class EntityNormalizer:
    def __init__(self):
        self.diagnostic_samples = []

    def normalize_unicode(self, text: str) -> str:
        """Decompose accents using NFKD and strip combining diacritics."""
        if not text:
            return ""
        decomposed = unicodedata.normalize('NFKD', text)
        return "".join(c for c in decomposed if not unicodedata.combining(c))

    def normalize_name(self, raw_name: str, record_diagnostic: bool = False) -> str:
        """Deterministic business name normalization."""
        if not raw_name:
            return ""
        
        orig = raw_name
        # 1. Unicode decomposition
        text = self.normalize_unicode(raw_name)
        
        # 2. Lowercase
        text = text.lower().strip()
        
        # 3. Strip URLs and domain extensions (.com, www., etc.)
        text = URL_REGEX.sub('', text)
        text = DOMAIN_SUFFIX_REGEX.sub('', text)
        
        # 4. Handle bracketed prefix (e.g. "[Corp] Dick Regional Armada" -> "Dick Regional Armada")
        m_bracket = BRACKETED_PREFIX_REGEX.match(text)
        if m_bracket:
            bracket_content = m_bracket.group(1).strip()
            rest = text[m_bracket.end():].strip()
            # If bracket contains legal word, strip it; otherwise append
            if bracket_content in STRICT_LEGAL_TERMS:
                text = rest
            else:
                text = f"{rest} {bracket_content}".strip()
                
        # 5. Replace punctuation with spaces
        text = re.sub(r'[\-_/\\,;:.\'"&|()\[\]{}+*#@!~?<>^%$`=]', ' ', text)
        
        # 6. Normalize whitespace
        tokens = text.split()
        if not tokens:
            return ""
            
        # 7. Strip leading or trailing strict legal terms
        # E.g. "llc crystal staffing" -> "crystal staffing", "ap hospitality inc" -> "ap hospitality"
        if len(tokens) > 1 and tokens[0] in STRICT_LEGAL_TERMS:
            tokens = tokens[1:]
        if len(tokens) > 1 and tokens[-1] in STRICT_LEGAL_TERMS:
            tokens = tokens[:-1]
        # Check if "pvt ltd" or "private limited" at the end
        if len(tokens) >= 2 and tokens[-2] in STRICT_LEGAL_TERMS and tokens[-1] in STRICT_LEGAL_TERMS:
            tokens = tokens[:-2]
            
        cleaned = " ".join(tokens)
        
        if record_diagnostic and len(self.diagnostic_samples) < 25:
            self.diagnostic_samples.append({"original": orig, "normalized": cleaned})
            
        return cleaned

    def extract_tokens(self, normalized_name: str, min_len: int = 2) -> list[str]:
        """Extract alphanumeric tokens from normalized name."""
        tokens = [t for t in normalized_name.split() if len(t) >= min_len and not t.isdigit()]
        return tokens

    def extract_compressed_core(self, normalized_name: str, prefix_len: int | None = None) -> str:
        """Remove all spaces to form a single continuous alphanumeric core string."""
        core = re.sub(r'\s+', '', normalized_name)
        if prefix_len is not None and prefix_len > 0:
            return core[:prefix_len]
        return core

    def extract_char_ngrams(self, normalized_name: str, n: int = 3) -> list[str]:
        """Generate character n-grams from normalized name."""
        # Pad with boundary tokens
        padded = f"#{normalized_name.strip()}#"
        if len(padded) < n:
            return [padded]
        return [padded[i:i+n] for i in range(len(padded) - n + 1)]

    def extract_address_signals(self, raw_addr: str, country: str) -> dict[str, str]:
        """Extract deterministic address features (postal/PIN code, state/region, street number)."""
        if not raw_addr:
            return {"postal_code": "", "region": "", "street_num": "", "is_empty": True}
        
        addr_clean = self.normalize_unicode(raw_addr).upper()
        postal_code = ""
        region = ""
        street_num = ""
        
        # 1. Postal/PIN code extraction (country generic)
        # 6-digit consecutive for India
        pins = re.findall(r'\b\d{6}\b', addr_clean)
        if pins:
            postal_code = pins[-1]
        else:
            # 5-digit for US / France
            zips = re.findall(r'\b\d{5}\b', addr_clean)
            if zips:
                postal_code = zips[-1]
                
        # 2. Street number (leading digits)
        m_num = re.match(r'^\s*(\d+)', addr_clean)
        if m_num:
            street_num = m_num.group(1)
            
        # 3. State/Region token (typically last comma-separated part or 2-letter US state)
        parts = [p.strip() for p in addr_clean.split(",") if p.strip()]
        if len(parts) >= 2:
            last_part = parts[-1].split()
            if last_part:
                region = last_part[0]
                
        return {
            "postal_code": postal_code,
            "region": region,
            "street_num": street_num,
            "is_empty": False
        }


# =====================================================================
# INVERTED INDEX DATA STRUCTURES
# =====================================================================

class InvertedPostingIndex:
    """Memory-efficient posting index storing 32-bit integer IDs partitioned by country."""
    def __init__(self, name: str = "Index"):
        self.name = name
        # Structure: dict[country, dict[key, array('I')]]
        self.index = defaultdict(lambda: defaultdict(lambda: array.array('I')))
        self.num_postings = 0
        self.num_keys = 0

    def add(self, country: str, key: str, int_id: int):
        if not key:
            return
        self.index[country][key].append(int_id)
        self.num_postings += 1

    def get_candidates(self, country: str, key: str) -> array.array:
        country_dict = self.index.get(country)
        if not country_dict:
            return array.array('I')
        return country_dict.get(key, array.array('I'))

    def finalize(self):
        """Compute key counts and statistics."""
        self.num_keys = sum(len(subdict) for subdict in self.index.values())

    def get_block_sizes(self, country: str | None = None) -> list[int]:
        """Return list of posting list lengths."""
        if country:
            return [len(postings) for postings in self.index.get(country, {}).values()]
        sizes = []
        for c_dict in self.index.values():
            sizes.extend(len(p) for p in c_dict.values())
        return sizes
