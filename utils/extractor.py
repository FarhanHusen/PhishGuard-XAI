import re
import tldextract
from urllib.parse import urlparse

def extract_features(url):
    features = {}
    
    # 1. Clean URL: Menangani input tanpa protokol agar tidak bias
    original_url = url.strip().lower()
    if not original_url.startswith(('http://', 'https://')):
        url_to_parse = 'https://' + original_url
    else:
        url_to_parse = original_url
        
    parsed = urlparse(url_to_parse)
    extracted = tldextract.extract(url_to_parse)
    
    # --- FITUR LEKSIKAL ---
    features['UrlLength'] = len(original_url)
    features['HostnameLength'] = len(parsed.netloc)
    # PathLength sekarang mengabaikan '/' tunggal agar tidak dianggap panjang
    path = parsed.path.strip('/')
    features['PathLength'] = len(path)
    
    features['NumDots'] = original_url.count('.')
    features['NumDash'] = original_url.count('-')
    features['NumNumericChars'] = sum(c.isdigit() for c in original_url)
    features['AtSymbol'] = 1 if '@' in original_url else 0
    
    # --- FITUR STRUKTURAL ---
    # HttpsInHostname: Phishing sering pakai "https-facebook-com.verify.id"
    features['HttpsInHostname'] = 1 if 'https' in parsed.netloc else 0
    
    # IsIpAddress: Cek apakah hostname cuma angka IP
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    features['IsIpAddress'] = 1 if re.match(ip_pattern, parsed.netloc) else 0
    
    # SubdomainLevel
    subdomain = extracted.subdomain
    features['SubdomainLevel'] = len(subdomain.split('.')) if subdomain else 0
    
    
    return features