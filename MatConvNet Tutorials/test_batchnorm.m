num_channels = 256;
height = 56; width = height;

bn_conv1_mult = net.params(6).value;
bn_conv1_bias = net.params(7).value;

bn_conv1_mean = net.params(8).value(:, 1);
bn_conv1_variance = net.params(8).value(:, 2);
epsilon = 0 * ones(height, width, num_channels);
variance = ones(height, width, num_channels);
mean = variance; bias = variance; mult = variance;

for channel = 1:num_channels
    mult(:, :, channel) = mult(:, :, channel) * bn_conv1_mult(channel, :);
    bias(:, :, channel) = bias(:, :, channel) * bn_conv1_bias(channel, :);
    mean(:, :, channel) = mean(:, :, channel) * bn_conv1_mean(channel, :);
    variance(:, :, channel) = variance(:, :, channel) * bn_conv1_variance(channel, :);
end

bn_tmp = mult .* ((net.vars(6).value - mean) ./ sqrt(variance .^ 2  + epsilon)) + bias;
bn_tmp2 = vl_nnbnorm(net.vars(6).value, bn_conv1_mult, bn_conv1_bias, 'epsilon', 1e-5, 'moments', net.params(8).value);
sum(sum(sum(bn_tmp - bn_tmp2)))
sum(sum(sum(net.vars(7).value - bn_tmp2)))