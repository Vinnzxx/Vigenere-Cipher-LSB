% Main Script
clc;
clear;
close all;

% Read the original image
originalImage = imread(['test25.jpg']);

% Convert the image to grayscale if it's not already
if size(originalImage, 3) > 1
    originalImage = rgb2gray(originalImage);
end

% Resize the image to 512x512 using the imresize function
desiredSize = [512, 512];
originalImage = imresize(originalImage, desiredSize);

% Flatten the image matrix into a 1D array for easier manipulation
imageData = originalImage(:);

% Encrypt a sample message using the Vigenere cipher with modulo 256
message = 'Hello, World!'; % Message with symbols and lowercase letters
key = 'SecretKey';         % Key with symbols and lowercase letters
encryptedMessage = vigenereEncryptMod256(message, key);
disp(['Encrypted Message: ', char(encryptedMessage)]);

% Embed the encrypted message in the LSB of the image pixels
numBits = 8; % Number of bits to replace (LSB)
modifiedImageData = imageData;
for i = 1:numel(encryptedMessage)
    bitsToReplace = de2bi(encryptedMessage(i), numBits, 'left-msb'); % Convert to binary
    for j = 1:numBits
        modifiedImageData(i) = bitset(modifiedImageData(i), j, bitsToReplace(j));
    end
end

% Reshape the modified 1D array back to the original image size
modifiedImage = reshape(modifiedImageData, size(originalImage));

% Save the modified image
imwrite(modifiedImage, 'modif.png');

% Retrieve the encrypted message from the LSB of the modified image
retrievedImageData = modifiedImage(:);
retrievedMessage = zeros(size(encryptedMessage));
bitIndex = 1;
for i = 1:numel(encryptedMessage)
    % Extract the next numBits bits from the retrieved image data
    bitsToRetrieve = bitget(retrievedImageData(i), 1:numBits);
    % Convert the extracted bits to a numeric value
    retrievedMessage(i) = bi2de(bitsToRetrieve, 'left-msb');
    % Update the bitIndex for the next iteration
    bitIndex = bitIndex + numBits;
end

% Decrypt the retrieved message using Vigenere decryption with modulo 256
decryptedMessage = vigenereDecryptMod256(retrievedMessage, key);

% Display the original and decrypted images side by side
figure;
subplot(2, 2, 1);
imshow(originalImage);
title('test1.jpg');

subplot(2, 2, 2);
imshow(modifiedImage);
title('Stego Image');

subplot(2, 2, 3);
imhist(originalImage);
title('Original Histogram');

subplot(2, 2, 4);
imhist(modifiedImage);
title('Stego Histogram');

% Display the original and decrypted messages
fprintf('Original Message: %s\n', message);
fprintf('Decrypted Message: %s\n', char(decryptedMessage));

% MSE
   originalImage = im2double(originalImage);
    modifiedImage = im2double(modifiedImage);
    mseValue = immse(originalImage, modifiedImage);
    %maxPixelValue = 1.0; % For images in the range [0, 1]
    %ssimValue = ssim(originalImage, originalImage);
    disp(['MSE: ', num2str(mseValue)]);
    %disp(['SSIM: ', num2str(ssimValue)]);
    
    % Calculate PSNR
MAX = 1.0; % Maximum pixel value for images in the range [0, 1]
psnrValue = 10 * log10((MAX^2) / mseValue);
disp(['PSNR: ', num2str(psnrValue), ' dB']);
    % SSIM
    % Parameters for SSIM calculation
    K1 = 0.01;
    K2 = 0.03;
    L = 1; % Dynamic range of pixel values (assumed to be 1 for images in the range [0, 1])
    % Compute mean, variance, and covariance of the original image
    mu1 = mean2(originalImage);
    sigma1_sq = var(originalImage(:));
    covariance = cov(originalImage(:), modifiedImage(:));
    sigma1 = sqrt(sigma1_sq);
    % Compute mean, variance, and covariance of the processed image
    mu2 = mean2(modifiedImage);
    sigma2_sq = var(modifiedImage(:));
    sigma2 = sqrt(sigma2_sq);
    % Compute the similarity components
    c1 = (K1 * L)^2;
    c2 = (K2 * L)^2;
    c3 = c2 / 2;
    % Compute SSIM components
    numerator = (2 * mu1 * mu2 + c1) * (2 * covariance(1, 2) + c2);
    denominator = (mu1^2 + mu2^2 + c1) * (sigma1_sq + sigma2_sq + c2);
    ssimValue = numerator / denominator;
    % Account for contrast masking effect
    contrastMask = (2 * sigma1 * sigma2 + c3) / (sigma1_sq + sigma2_sq + c3);
    ssimValue = ssimValue * contrastMask;
    % Display the result
    disp(['SSIM: ', num2str(ssimValue)]);
    
%     % BER
%     originalBinary = originalImage >= 0.5;
%     processedBinary = modifiedImage >= 0.5;
%     numErrors = sum(originalBinary(:) ~= processedBinary(:));
%     totalPixels = numel(originalImage);
%     ber = numErrors / totalPixels;
%     disp(['BER: ', num2str(ber)]);
% Convert the decrypted message back to binary (thresholding with 0.5)
    decryptedBinary = uint8(decryptedMessage) >= 128;
    
    % Convert the original message to binary (thresholding with 0.5)
    originalBinary = uint8(message) >= 128;
    
    % Calculate the number of bit errors
    numErrors = sum(decryptedBinary(:) ~= originalBinary(:));
    
    % Calculate the Bit Error Rate (BER)
    ber = numErrors / numel(originalBinary);
    
    % Display the BER
    disp(['BER: ', num2str(ber)]);
    
    % Entropy original
    % Compute the histogram of the grayscale image
    histogram1 = imhist(originalImage);
    % Calculate the total number of pixels in the image
    totalPixels = numel(originalImage);
    % Compute the probability of occurrence for each intensity level
    probability = histogram1 / totalPixels;
    % Calculate the entropy using the probability values
    entropy_value = -sum(probability .* log2(max(1e-10, probability)));
    % Display the result
    disp(['Entropy original: ', num2str(entropy_value)]);
    
    % Entropy encrypt
    num_pixels = numel(modifiedImage);
    pixel_counts = imhist(modifiedImage); % Compute the histogram of pixel intensities
    probability_distribution = pixel_counts / num_pixels;

    entropy_value = -sum(probability_distribution .* log2(probability_distribution + eps)); % Add 'eps' to avoid log(0)

    disp(['Entropy Encrypted: ', num2str(entropy_value)]);
