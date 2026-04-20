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
    hostname = parsed.netloc
    path = parsed.path.strip('/')
    
    # --- FITUR LEKSIKAL ---
    features['UrlLength'] = len(original_url)
    features['HostnameLength'] = len(hostname)
    features['PathLength'] = len(path)
    
    features['NumDots'] = original_url.count('.')
    features['NumDash'] = original_url.count('-')
    features['NumNumericChars'] = sum(c.isdigit() for c in original_url)
    features['AtSymbol'] = 1 if '@' in original_url else 0
    
    # --- FITUR STRUKTURAL (LEBIH CANGGIH) ---
    
    # 1. HttpsInHostname: Deteksi "https-login.com"
    features['HttpsInHostname'] = 1 if 'https' in hostname else 0
    
    # 2. IsIpAddress: Deteksi penggunaan IP sebagai domain
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    features['IsIpAddress'] = 1 if re.match(ip_pattern, hostname) else 0
    
    # 3. SubdomainLevel
    subdomain = extracted.subdomain
    features['SubdomainLevel'] = len(subdomain.split('.')) if subdomain else 0
    
    # 4. FITUR BARU: Abnormal Subdomain (Kunci untuk membedakan UNISA vs Phishing)
    # Phishing sering punya kata "login", "verify", "secure" di subdomain
    sensitive_words = ['login', 'verify', 'secure', 'update', 'banking', 'account']
    features['AbnormalSubdomain'] = 1 if any(word in subdomain for word in sensitive_words) else 0

    # 5. FITUR BARU: IsAcademicDomain (Membantu deteksi .ac.id agar aman)
    # Memberikan sinyal kuat ke AI bahwa ini institusi pendidikan
    features['IsAcademicDomain'] = 1 if extracted.suffix == 'ac.id' or extracted.suffix == 'edu' else 0

    # 6. FITUR BARU: NumSensitiveWords (Deteksi kata pancingan di seluruh URL)
    features['NumSensitiveWords'] = sum(1 for word in sensitive_words if word in original_url)

    return features