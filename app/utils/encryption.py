from cryptography.fernet import Fernet
from ..config import get_settings
import base64
import hashlib

settings = get_settings()

def get_encryption_key():
    # Use the secret key to generate a Fernet key
    key = hashlib.sha256(settings.SECRET_KEY.encode()).digest()
    return base64.urlsafe_b64encode(key)

def encrypt_filename(filename: str) -> str:
    """Encrypt a filename using Fernet symmetric encryption."""
    f = Fernet(get_encryption_key())
    encrypted = f.encrypt(filename.encode())
    return encrypted.decode()

def decrypt_filename(encrypted_filename: str) -> str:
    """Decrypt an encrypted filename."""
    f = Fernet(get_encryption_key())
    decrypted = f.decrypt(encrypted_filename.encode())
    return decrypted.decode() 