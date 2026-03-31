from werkzeug.security import generate_password_hash, check_password_hash

while(True):
    print(check_password_hash("scrypt:32768:8:1$EqEnTLZDefvAmiC7$4706c61fde66a2815901fbc1d7a009302bfdbee64810b8d6423da3b06ead826207365b908eb8753463315520e5204d3b397b788cbf135b5bccc68d5a5ebde8f5", input()))