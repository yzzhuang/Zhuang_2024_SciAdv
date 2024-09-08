function out = qdm_z(obs, mod_cp, var, frq, trace, jitter_factor)
%QDM_Z Performs Quantile Delta Mapping (QDM) bias correction.
%   This function performs QDM bias correction and was partially based on 
%   ClimQMBC (MATLAB) and ClimDown (R), optimized for efficiency.
%   The code involving moving projected model windows is adapted from ClimQMBC.
%   The core code for non-parametric distribution fitting and correction 
%   is adapted from ClimDown (which uses parametric fitting).
%
%   This function requires Statistics and Machine Learning Toolbox and the
%   Parallel Computing Toolbox from MATLAB. To use the function without 
%   parallelization, replace all "parfor" with "for".
%
% INPUT:
%   obs           - Observed data (nlon x nlat x nt_c); nt_c: length of 
%                   historical period (reference for correction). `obs` 
%                   should start in January of the first year and end in 
%                   December of the last year (if monthly).
%
%   mod_cp        - Modeled data (nlon x nlat x nt_cp); nt_cp = nt_c + nt_p,
%                   length of the model period. The first nt_c should 
%                   represent the same time period as the observation. 
%                   `mod_cp` should start in January of the first year and 
%                   end in December of the last year (if monthly).
%
%   var           - A flag that identifies the data type.
%                   - Temperature or usually non-bounded data: var = 0 
%                     (correction is done by addition/subtraction)
%                   - Precipitation or usually bounded or positive-only 
%                     data: var = 1 (correction is done by multiplication/division)
%
%   frq           - A character specifying if the input is annual or monthly 
%                   data. If not specified, it is set to monthly as default.
%                   - Monthly:    frq = 'M'
%                   - Annual:     frq = 'A' (not tested)
% 
%   trace         - A float indicating the threshold to consider 
%                   physically null precipitation values (only when var == 1).
%
%   jitter_factor - A float indicating the maximum value of the random 
%                   values added to original data to accommodate ties.
%
% OUTPUT:
%   out           - QDM corrected modeled data (same size as `mod_cp`).
%
% REFERENCES:
%   Cannon, A. J., et al., 2015: Bias correction of GCM precipitation by 
%   quantile mapping: How well do methods preserve changes in quantiles 
%   and extremes? J. Climate, 28(17), 6938-6959,
%   https://doi.org/10.1175/JCLI-D-14-00754.1
%
%   ClimDown: https://github.com/pacificclimate/ClimDown
%   ClimQMBC: https://github.com/saedoquililongo/climQMBC
%
% Yizhou Zhuang, UCLA
% May 15, 2024
%% Set default values for optional arguments if not provided
if ~exist('trace','var')
    trace = 0.05;
end

if ~exist('jitter_factor','var')
    jitter_factor = 0.01;
end

if ~exist('frq', 'var')
    frq = 'M'; % Default frequency set to monthly ('M')
end

if strcmp(frq, 'M')
    I = 12; % Monthly data
elseif strcmp(frq, 'Y')
    I = 1; % Yearly data
else
    error('undefined frq option!');
end
%% Preprocessing
% Check if input data (observation or model) contains only NaN values
if all(isnan(mod_cp), 'all') || all(isnan(obs), 'all')
    out = nan(size(mod_cp)); % Return NaN output if input data is all NaN
    return;
end

% Verify that the spatial dimensions of observed and modeled data match
N_obs = size(obs); N_mod = size(mod_cp); 
if N_obs(1)~=N_mod(1) || N_obs(2)~=N_mod(2)  
    error('Spatial dimensions do not match between observation and model data!');
end

% Remove grid point with all NaN data
n_ll = N_obs(1) * N_obs(2);   % Total number of grid points
mask = ~(all(isnan(obs), 3:ndims(obs)) | all(isnan(mod_cp), 3:ndims(mod_cp)));  % Mask for valid grid points
n_llm = sum(mask(:)); 
N = n_llm * I;

% Reshape and filter the data based on the mask
obs = reshape(double(obs), n_ll, I, []);
mod_cp = reshape(double(mod_cp), n_ll, I, []);
obs = reshape(obs(mask(:), :, :), N, []); 
ny_obs = size(obs, 2); % Number of years in observed data
mod_cp = reshape(mod_cp(mask(:), :, :), N, []); 
ny_mod = size(mod_cp, 2); % Number of years in modeled data

% Apply jitter to handle ties 
obs = obs + jitter_factor * rand(size(obs));
mod_cp = mod_cp + jitter_factor * rand(size(mod_cp));

% Handle precipitation data for zero values as left censored
if var == 1
    epsilon = eps('double'); % Smallest positive number in double precision
    ind = obs < trace & ~isnan(obs); % Identify low values in observed data
    obs(ind) = epsilon + (trace-epsilon) * rand(sum(ind(:)), 1); % Adjust low values
    ind = mod_cp < trace & ~isnan(mod_cp); % Identify low values in modeled data
    mod_cp(ind) = epsilon + (trace-epsilon) * rand(sum(ind(:)), 1); % Adjust low values
end
%% Bias Correction
% Calculate quantiles for observed and modeled data during the reference period
tau = (1 : ny_obs)/(ny_obs + 1); % Quantile levels
quant_obs = quantile(obs, tau, 2); % Quantiles for observed data
mod_c = mod_cp(:,1:ny_obs); % Modeled data during the reference period
quant_mod_c = quantile(mod_c, tau, 2); % Quantiles for modeled data

% Perform bias correction for the model during the reference period
mhat_c = nan(N, ny_obs);
parfor i = 1 : N
    % ind = ~isnan(mod_c(i,:));
    % if ~isempty(ind)
    %     F = griddedInterpolant(quant_mod_c(i,:), quant_obs(i,:), 'linear', 'nearest');
    %     mhat_c(i,ind) = F(mod_c(i,ind));
    % end
    F = griddedInterpolant(quant_mod_c(i,:), quant_obs(i,:), 'linear', 'nearest');
    mhat_c(i,:) = F(mod_c(i,:));
end

% Bias correction for the projected model period  
if var == 1
    F1 = griddedInterpolant(tau', quant_mod_c', 'linear', 'nearest');
    F2 = griddedInterpolant(tau', quant_obs', 'linear', 'nearest');
else
    FD = griddedInterpolant(tau', quant_obs'-quant_mod_c', 'linear', 'nearest');
end

% Construct projected windows with the same length as the reference period
ind = (1:ny_obs)' + (1:ny_mod-ny_obs);
mod_P = reshape(mod_cp(:,ind(:)), N, ny_obs, []); % Projected model data (window)
quant_mod_P = quantile(mod_P, tau, 2);  % Quantiles for projected model data (window)
mod_P1 = squeeze(mod_P(:,end,:));  % Latest year of window

adj = nan(N, ny_mod-ny_obs);
if var == 1
    parfor j = 1 : ny_mod-ny_obs
        F_mod_p1 = nan(N,1);
        for i = 1 : N
            F = griddedInterpolant(quant_mod_P(i,:,j), tau, 'linear', 'nearest');
            F_mod_p1(i) = F(mod_P1(i,j));
        end
        adj(:,j) = diag(F2(F_mod_p1)) ./ diag(F1(F_mod_p1));
    end
    mhat_p = mod_P1 .* adj; % Apply multiplicative adjustment
else
    parfor j = 1 : ny_mod-ny_obs
        F_mod_p1 = nan(N,1);
        for i = 1 : N
            F = griddedInterpolant(quant_mod_P(i,:,j), tau, 'linear', 'nearest');
            F_mod_p1(i) = F(mod_P1(i,j));
        end
        adj(:,j) = diag(FD(F_mod_p1));
    end
    mhat_p = mod_P1 + adj; % Apply additive adjustment
end

% Combine bias-corrected data from reference and projected periods
mhat = [mhat_c, mhat_p];
if var == 1
    mhat(mhat < trace) = 0;
end

% Reshape the output data to the original dimensions
out = nan(n_ll, I, ny_mod);
out(mask(:),:,:) = reshape(mhat, n_llm, I, []);
out = reshape(out, N_mod);
end