% Vigenere Decryption Function with Modulo 256
function decryptedMsg = vigenereDecryptMod256(encryptedMsg, key)
    % Convert numeric array to uint8
    encryptedMsg = uint8(encryptedMsg);
    key = uint8(key);

    % Repeat the key to match the length of the encrypted message
    keyRepeats = ceil(length(encryptedMsg) / length(key));
    expandedKey = repmat(key, 1, keyRepeats);
    expandedKey = expandedKey(1:length(encryptedMsg));

    % Decrypt the message using Vigenere cipher with modulo 256
    decryptedMsg = mod(int16(encryptedMsg) - int16(expandedKey), 256);
end
