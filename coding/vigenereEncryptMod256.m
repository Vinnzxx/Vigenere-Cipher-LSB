% Vigenere Encryption Function with Modulo 256
function encryptedMsg = vigenereEncryptMod256(plainText, key)
    % Convert text to numeric array using ASCII values
    plainText = uint8(plainText);
    key = uint8(key);

    % Repeat the key to match the length of the plaintext
    keyRepeats = ceil(length(plainText) / length(key));
    expandedKey = repmat(key, 1, keyRepeats);
    expandedKey = expandedKey(1:length(plainText));

    % Encrypt the message using Vigenere cipher with modulo 256
    encryptedMsg = mod(plainText + expandedKey, 256);
end