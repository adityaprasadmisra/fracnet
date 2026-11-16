function [y,w]=add_awgn(x,SNR_dB,reference_power)
% Add circular complex AWGN; preserve input dimensions.
if nargin<3 || isempty(reference_power), reference_power=mean(abs(x(:)).^2); end
if reference_power<=0, error('Signal/reference power must be positive.'); end
np=reference_power/10^(SNR_dB/10);
w=sqrt(np/2)*(randn(size(x))+1j*randn(size(x)));
y=x+w;
end
