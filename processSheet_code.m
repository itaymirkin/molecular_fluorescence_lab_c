filename = 'results.xlsx';  % Update this to your actual file name

% Get sheet names
[~, sheets] = xlsfinfo(filename);

% Columns to select
dataColumns = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29];
timeColumns = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30];
concentrations = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0008, 0.0005, 0.0001];

smallConcentrations = [0.0001, 0.0005, 0.0008, 0.001];

% Initialize figures for combined plots
figCombinedIntegrated = figure('Position', [100, 100, 800, 600]);
figCombinedLinearFit = figure('Position', [100, 100, 800, 600]);

% Color map for different sheets
colors = {'r', 'g', 'b'};

% Process each sheet
for j = 1:3
    sheetName = sheets{j};
    disp(['Processing sheet: ', sheetName]);
    
    % Read XLSX sheet data
    try
        rawData = readtable(filename, 'Sheet', sheetName, 'VariableNamingRule', 'preserve');
        disp(['Successfully read sheet: ', sheetName]);
    catch ME
        warning('Error reading sheet %s: %s', sheetName, ME.message);
        continue;  % Skip to the next sheet
    end
    
    % Select data and time columns
    dataTable = rawData(:, dataColumns);
    timeTable = rawData(:, timeColumns);
    
    % Print debug information about the tables
    disp(['dataTable size: ', num2str(size(dataTable))]);
    disp(['timeTable size: ', num2str(size(timeTable))]);
    
    try
        results = processSheet(dataTable, timeTable, sheetName, concentrations);
        %writetable(results,'excel_result_new.xlsx','Sheet',sheetName)
        % Plot for integrated results of this sheet on the combined figure
        figure(figCombinedIntegrated);
        hold on;
        errorbar(flipud(concentrations), results.NormalizedIntensity, results.NormalizedIntensityError, '-o', 'Color', colors{j}, 'DisplayName', sheetName);
        errorbar(flipud(concentrations), results.NormalizedIntensity, 0.05*flipud(concentrations),'horizontal', 'Color', colors{j} ,'HandleVisibility', 'off');
        % Plot linear fit for small concentrations on the combined linear fit figure
        figure(figCombinedLinearFit);
        hold on;
        smallConcIdx = ismember(concentrations, smallConcentrations);
        smallConc = concentrations(smallConcIdx);
        smallIntensities = results.NormalizedIntensity(smallConcIdx);
        errorbar(smallConc,smallIntensities, results.NormalizedIntensityError(smallConcIdx), '.', 'Color', colors{j}, 'HandleVisibility', 'off');
        errorbar(smallConc, smallIntensities, 0.05*flipud(smallConc),'horizontal', '.', 'Color', colors{j} ,'HandleVisibility', 'off');
        [p, S] = polyfit(smallConc, smallIntensities, 1);
        yfit = polyval(p, smallConc);
        %plot(smallConc, smallIntensities, 'o', 'Color', colors{j}, 'DisplayName', [sheetName ' Data']);
      
        plot(smallConc, yfit, '-', 'Color', colors{j}, 'LineWidth', 2, 'DisplayName', [sheetName ' Fit']);
        
        % Print the values of the linear fit
        fprintf('Linear fit for small concentrations in sheet %s:\n', sheetName);
        fprintf('Slope: %f, Intercept: %f\n', p(1), p(2));
        slope = p(1);
        
        % Get the diagonal elements of the covariance matrix, which represent the variances
        cov_matrix = S.normr^2 * inv(S.R) * inv(S.R)';
        var_b = cov_matrix(1,1); % Variance of the slope
        var_a = cov_matrix(2,2); % Variance of the intercept

        % Standard errors
        std_error_slope = sqrt(var_b);
        std_error_intercept = sqrt(var_a);
        
        epsilon = slope ;
        fprintf('Molar Attenuation Coefficient (epsilon): %f (path length: %f cm)\n', epsilon, 1);
        disp(['Standard error of the slope: ', num2str(std_error_slope)]);
        disp(['Standard error of the intercept: ', num2str(std_error_intercept)]);
    catch ME
        warning('Error processing sheet %s: %s', sheetName, ME.message);
    end
end

% Finalize the combined integrated intensity plot
figure(figCombinedIntegrated);
title('Normalized Integrated Intensity for Wavelengths > 556 nm (All Sheets)');
xlabel('Concentration');
ylabel('Normalized Integrated Intensity');
set(gca, 'XScale', 'linear');
xlim([0 0.1])
%xticks([0.0001, 0.0005, 0.0008, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]);
%xticklabels({'0.0001', '0.0005', '0.0008', '0.001', '0.0025', '0.005', '0.01', '0.025', '0.05', '0.1'});
xticks([ 0.001,  0.01, 0.05, 0.1]);
xticklabels({'0.001',  '0.01', '0.05', '0.1'});
legend('Location', 'best');
grid on;
hold off;

% Save the combined integrated intensity plot
saveas(figCombinedIntegrated, 'combined_integrated_intensity_plot.png');

% Finalize the combined linear fit plot
figure(figCombinedLinearFit);
title('Linear Fits for Small Concentrations (All Sheets)');
xlabel('Concentration');
ylabel('Normalized Integrated Intensity');%
% set(gca, 'XScale', 'log');
xticks(smallConcentrations);
xticklabels(cellstr(num2str(smallConcentrations', '%.4f')));
legend('Location', 'best');
grid on;
hold off;

% Save the combined linear fit plot
saveas(figCombinedLinearFit, 'combined_linear_fit_plot.png');

disp('Processing complete. Combined plots have been saved.');


% Function to process a single sheet
function results = processSheet(dataTable, timeTable, sheetName, concentrations)
    results = table();
    numSamples = width(dataTable) / 2;  % Number of samples in this sheet
    
    % Define a constant relative error (e.g., 5%)
    relativeError = 0.05; 

    for sample = 1:numSamples
        try
            disp(['Processing sample ', num2str(sample)]);
            wavelengthCol = dataTable{:, 2*sample - 1};  % X [nm] column
            intensityCol = dataTable{:, 2*sample};       % Y [Intensity] column
            time = timeTable{1, sample};                 % Time [ms] value
            conc = concentrations(sample);
            
            % Print debug information about the current sample
            disp(['wavelengthCol size: ', num2str(size(wavelengthCol))]);
            disp(['intensityCol size: ', num2str(size(intensityCol))]);
            disp(['time: ', num2str(time)]);
            
            % Convert to numeric and remove any non-numeric values
            wavelengths = str2double(string(wavelengthCol));
            intensities = str2double(string(intensityCol));

            validIdx = ~isnan(wavelengths) & ~isnan(intensities);
            wavelengths = wavelengths(validIdx);
            intensities = intensities(validIdx);

            % Ensure we have vectors
            wavelengths = wavelengths(:);
            intensities = intensities(:);

            % Calculate errors (assuming constant relative error)
            intensityErrors = relativeError * intensities;

            % Normalize raw intensities by time
            normalizedIntensities = intensities / time;
            normalizedIntensityErrors = intensityErrors / time;

            % Apply moving average filter to smooth the data
            windowSize = 5;  % Adjust the window size as needed
            smoothedNormalizedIntensities = movmean(normalizedIntensities, windowSize);
            smoothedNormalizedIntensityErrors = movmean(normalizedIntensityErrors, windowSize);
    
            % Find indices where wavelength > 556
            mask = wavelengths > 460;

            if any(mask)
                % Integrate intensity for wavelengths > 556
                integratedIntensity = abs(trapz(wavelengths(mask), intensities(mask)));
                integratedIntensityError = abs(trapz(wavelengths(mask), intensityErrors(mask)));
                
                % Normalize by maximum integrated intensity
                normalizedIntensity = integratedIntensity / time;
                normalizedIntensityError = integratedIntensityError / time;
            else
                integratedIntensity = 0;  % No wavelengths above 556
                normalizedIntensity = 0;
                normalizedIntensityError = 0;
            end

            % Store results
            results = [results; table(sample, integratedIntensity, normalizedIntensity, integratedIntensityError, normalizedIntensityError, {wavelengths}, {normalizedIntensities}, {smoothedNormalizedIntensities}, {smoothedNormalizedIntensityErrors}, {sheetName}, {conc}, 'VariableNames', {'Sample', 'IntegratedIntensity', 'NormalizedIntensity', 'IntegratedIntensityError', 'NormalizedIntensityError', 'Wavelength', 'NormalizedIntensityRaw', 'SmoothedNormalizedIntensityRaw', 'SmoothedNormalizedIntensityRawError', 'SheetName', 'concentration'})];
            disp('here')
        catch ME
            warning('Error processing sample %d in sheet %s: %s', sample, sheetName, ME.message);
        end
    end
    
    % Additional debug output after the loop
    disp('Completed sample processing');
    disp(['results size: ', num2str(size(results))]);
    disp('First few rows of results:');
    disp(head(results));
end

