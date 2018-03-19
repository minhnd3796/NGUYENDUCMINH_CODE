num_channels = 64;

bn_conv1_mult = net.params(2).value;
bn_conv1_bias = net.params(3).value;

bn_conv1_mean = net.params(4).value(:, 1);
bn_conv1_variance = net.params(4).value(:, 2);
epsilon = 1e-5 * ones(112, 112, 64);

variance = ones(112, 112, 64);
mean = variance; bias = variance; mult = variance;

for channel = 1:64
    mult(:, :, channel) = mult(:, :, channel) * bn_conv1_mult(channel, :);
    bias(:, :, channel) = bias(:, :, channel) * bn_conv1_bias(channel,:);
    mean(:, :, channel) = mean(:, :, channel) * bn_conv1_mean(channel,:);
    variance(:, :, channel) = variance(:, :, channel) * bn_conv1_variance(channel,:);
end

conv1x = mult .* ((conv1 - mean) ./ sqrt(variance .^2  + epsilon)) + bias;