import secrets
import string

def generate_password(length=16, uppercase=True, lowercase=True, digits=True, symbols=True):
    """
    Generates a secure random password with specified character types.

    Args:
        length (int): The desired length of the password. Must be at least 4.
        uppercase (bool): Whether to include uppercase letters.
        lowercase (bool): Whether to include lowercase letters.
        digits (bool): Whether to include digits.
        symbols (bool): Whether to include symbols.

    Returns:
        str: A securely generated random password.
        
    Raises:
        ValueError: If no character types are selected or length is too short.
    """
    char_pool = ''
    password_chars = []

    if uppercase:
        char_pool += string.ascii_uppercase
        password_chars.append(secrets.choice(string.ascii_uppercase))
    if lowercase:
        char_pool += string.ascii_lowercase
        password_chars.append(secrets.choice(string.ascii_lowercase))
    if digits:
        char_pool += string.digits
        password_chars.append(secrets.choice(string.digits))
    if symbols:
        char_pool += string.punctuation
        password_chars.append(secrets.choice(string.punctuation))

    if not char_pool:
        raise ValueError("At least one character type must be selected.")

    if length < len(password_chars):
        raise ValueError(f"Length must be at least {len(password_chars)} to include all selected character types.")

    # Fill the rest of the password with random characters from the entire pool
    remaining_length = length - len(password_chars)
    for _ in range(remaining_length):
        password_chars.append(secrets.choice(char_pool))

    # Shuffle the list to ensure the guaranteed characters are not in a predictable order
    secrets.SystemRandom().shuffle(password_chars)

    return ''.join(password_chars)

if __name__ == '__main__':
    print("--- Secure Password Generator ---")
    
    # Example 1: Default password
    default_password = generate_password()
    print(f"Default (length 16, all types): {default_password}")

    # Example 2: No symbols
    no_symbols_password = generate_password(length=12, symbols=False)
    print(f"Length 12 (no symbols):       {no_symbols_password}")

    # Example 3: Digits and letters only
    letters_digits_password = generate_password(length=20, uppercase=True, lowercase=True, digits=True, symbols=False)
    print(f"Length 20 (letters & digits): {letters_digits_password}")

    # Example 4: Lowercase only (will raise error if length is too small, but fine for 8)
    try:
        lowercase_only = generate_password(length=8, uppercase=False, digits=False, symbols=False)
        print(f"Length 8 (lowercase only):      {lowercase_only}")
    except ValueError as e:
        print(f"Error generating lowercase-only password: {e}")
