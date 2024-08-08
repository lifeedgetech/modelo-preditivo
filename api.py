import secrets

def generate_api_key():
    return secrets.token_hex(16)

def save_api_key(api_key):
    with open('api_key.txt', 'w') as f:
        f.write(api_key)

if __name__ == '__main__':
    api_key = generate_api_key()
    save_api_key(api_key)
    print(f'API Key generated and saved: {api_key}')
