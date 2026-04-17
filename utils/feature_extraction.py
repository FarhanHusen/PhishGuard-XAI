import re
from urllib.parse import urlparse

def extract_features(url):
    parsed = urlparse(url)

    features = {}

    features["NumDots"] = url.count('.')
    features["UrlLength"] = len(url)
    features["NumDash"] = url.count('-')
    features["AtSymbol"] = 1 if '@' in url else 0
    features["IpAddress"] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    features["HttpsInHostname"] = 1 if "https" in parsed.netloc else 0
    features["PathLevel"] = parsed.path.count('/')
    features["PathLength"] = len(parsed.path)
    features["NumNumericChars"] = sum(char.isdigit() for char in url)

    return features