% Define the list of image files
image_files = {'Fe1.jpeg', 'Fe2.jpeg', 'Fe3.jpeg', 'Fe4.jpeg', 'Fe5.jpeg', ...
               'Fe6.jpeg', 'Fe7.jpeg', 'Fe8.jpeg', 'Fe9.jpeg', 'Fe10.jpeg'};

slope_vec = zeros(length(image_files));
% Main analysis loop
for i = 1:length(image_files)
    % Get or load line data
    [x_line, y_line, line_width] = getLineData(image_files{i});

    % Read and process the image
    A = imread(image_files{i});  
    A1 = im2double(A); 
    [height , width, ~] = size(A1);

    % Extract intensity values along the line
    intensity = zeros(1, length(x_line));
    for j = 1:length(x_line)
        % Calculate the coordinates for the averaging
        x_start = max(1, round(x_line(j) - line_width / 2));
        x_end = min(width, round(x_line(j) + line_width / 2));
        y_start = max(1, round(y_line(j) - line_width / 2));
        y_end = min(height, round(y_line(j) + line_width / 2));

        % Average intensity over the width
        region = A1(y_start:y_end, x_start:x_end, 2);  % Using green channel
        intensity(j) = mean(region(:));
    end

    % Convert pixel distances to cm (assuming 10 cm across the full image width)
    x_cm = (x_line - min(x_line)) * 10/(max(x_line) - min(x_line));

    % Create a new figure for analysis
    figure('Name', ['Fe ' num2str(i) ' Analysis'], 'NumberTitle', 'off');

    % Plot the image with the selected line
    figure();
    imagesc(A1(:,:,2));
    hold on;
    plot(x_line, y_line, 'r-', 'LineWidth', 2);
    title(['Fe ' num2str(i) ' - Selected Line']);
    saveas(gcf, ['Fe_' num2str(i) '_image_filter.png']);
    hold off
    % Plot the intensity profile
    figure();
    hold on
    plot(x_cm, intensity);
    title(['Fe ' num2str(i) ' - Intensity Profile']);
    xlabel('Distance [cm]');
    ylabel('Intensity');
    grid minor;
    saveas(gcf, ['Fe_' num2str(i) '_image_profile.png']);
    hold off
    % Calculate and plot log of the intensity, and perform linear fit
    
    log_intensity = log(intensity/max(intensity));
    figure();
    scatter(x_cm, log_intensity);
    hold on;

    % Perform linear fit
    [p, S] = polyfit(x_cm, log_intensity, 1);
    y_fit = polyval(p, x_cm);
    plot(x_cm, y_fit, 'r-', 'LineWidth', 2);
    disp(['slope: ' num2str(p(1))])
    slope_vec(i) = p(1);  
    xlabel('Distance [cm]');
    ylabel('log(Intensity)');
    title(['Fe ' num2str(i) ' - Log Intensity and Fit']);
    legend('Log(Data)', 'Linear Fit');
    grid minor;
    annotation('textbox', [0.74, 0.6, 0.1, 0.1], 'String', ...
        sprintf('Fit parameters:\nSlope: %f\nIntercept: %f\nR-squared: %f', ...
        p(1), p(2), 1 - (S.normr/norm(log_intensity - mean(log_intensity)))^2), ...
        'EdgeColor', 'none');
    saveas(gcf, ['Fe_' num2str(i) '_image_fit.png']);
    % Display fit parameters on the figure
    
    hold off;

    % Calculate and plot residuals
    residuals = log_intensity - y_fit;
    figure();
    scatter(x_cm, residuals);
    xlabel('Distance [cm]');
    ylabel('Residuals');
    yline(0, 'r');
    title(['Fe ' num2str(i) ' - Residuals']);
    grid minor;
    saveas(gcf, ['Fe_' num2str(i) '_image_residuals.png']);
    hold off;
    
  

    % Adjust figure size and layout
    set(gcf, 'Position', get(0, 'Screensize')); % Maximize figure window

    % Optionally save the figure
     saveas(gcf, ['Fe_' num2str(i) '_analysis.png']);

    % Wait for user to press a key before moving to the next image
%     disp('Press any key to continue to the next image...');
%     pause;
end
% Function to get or load line data
function [x_line, y_line, line_width] = getLineData(image_file)
    data_file = [image_file(1:end-5) '_line_data.mat'];
    
    if exist(data_file, 'file')
        % Load existing data
        load(data_file, 'x_line', 'y_line', 'line_width');
        if ~exist('line_width', 'var')
            % If line_width is not in the file, prompt the user
            disp('Line width not found in existing data. Please enter it now.');
            line_width = input('Enter the width (in pixels) for averaging intensity: ');
            % Save the updated data
            save(data_file, 'x_line', 'y_line', 'line_width');
        else
            disp(['Loaded existing line data for ' image_file]);
        end
    else
        % File does not exist, prompt the user to select line and width
        % Read the image
        A = imread(image_file);
        [height, width, ~] = size(A);

        % Display the image and let the user select a line
        figure('Name', ['Select Line for ' image_file], 'NumberTitle', 'off');
        imagesc(A(:,:,2));
        title(['Select two points to define the line of interest for ' image_file]);

        % Use ginput to get two points from the user
        [x, y] = ginput(2);

        % Calculate the line parameters
        slope = (y(2) - y(1)) / (x(2) - x(1));
        intercept = y(1) - slope * x(1);

        % Generate points along the line
        x_line = min(x):max(x);
        y_line = slope * x_line + intercept;

        % Round to nearest integer (pixel coordinates)
        x_line = round(x_line);
        y_line = round(y_line);

        % Ensure coordinates are within image bounds
        x_line = max(1, min(x_line, width));
        y_line = max(1, min(y_line, height));

        % Prompt user to input the width for averaging
        line_width = input('Enter the width (in pixels) for averaging intensity: ');

        % Save the line data
        save(data_file, 'x_line', 'y_line', 'line_width');
        disp(['Saved new line data for ' image_file]);

        % Close the line selection figure
        close;
    end
end
