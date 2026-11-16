function report = check_revision20_installation()
% CHECK_REVISION20_INSTALLATION
% Verify that MATLAB resolves the intended Revision 20 Person 2 files.
% Run this from the Person 2 folder before test_channel_estimator.

    thisFile = mfilename('fullpath');
    person2Dir = fileparts(thisFile);
    originalDir = pwd;
    restoreDirectory = onCleanup(@() cd(originalDir)); %#ok<NASGU>

    cd(person2Dir);
    pathEntries = strsplit(path,pathsep);
    normalizePath = @(p) lower(regexprep(strtrim(char(p)),'[\\/]+$',''));
    normalizedPathEntries = cellfun(normalizePath,pathEntries, ...
        'UniformOutput',false);
    if any(strcmp(normalizePath(person2Dir),normalizedPathEntries))
        rmpath(person2Dir);
    end
    addpath(person2Dir,'-begin');
    rehash;

    functionNames = { ...
        'system_parameters', ...
        'fractional_doppler_dictionary', ...
        'fractional_doppler_coarse_estimation', ...
        'apply_fractional_doppler_channel', ...
        'evaluate_channel_parameter_estimation'};
    expectedInputs = [0 4 8 3 7];
    expectedOutputs = [1 3 5 1 1];

    report.function = functionNames(:);
    report.path = cell(numel(functionNames),1);
    report.nargin = zeros(numel(functionNames),1);
    report.nargout = zeros(numel(functionNames),1);
    report.ok = false(numel(functionNames),1);

    allOK = true;
    fprintf('\nRevision 20 installation check\n');
    fprintf('Person 2 folder: %s\n\n',person2Dir);
    for k = 1:numel(functionNames)
        name = functionNames{k};
        expectedFile = fullfile(person2Dir,[name '.m']);
        actualFile = which(name);
        report.path{k} = actualFile;
        if isempty(actualFile)
            fprintf('FAIL  %s: not found on MATLAB path.\n',name);
            allOK = false;
            continue;
        end
        try
            report.nargin(k) = nargin(name);
            report.nargout(k) = nargout(name);
        catch
            fprintf('FAIL  %s: MATLAB resolved a script or non-callable file: %s\n', ...
                name,actualFile);
            allOK = false;
            continue;
        end
        samePath = strcmpi(strrep(actualFile,'/','\'), ...
            strrep(expectedFile,'/','\'));
        sameSignature = report.nargin(k)==expectedInputs(k) && ...
            report.nargout(k)==expectedOutputs(k);
        report.ok(k) = samePath && sameSignature;
        if report.ok(k)
            fprintf('PASS  %-42s nargin=%d nargout=%d\n', ...
                name,report.nargin(k),report.nargout(k));
        else
            fprintf('FAIL  %s\n',name);
            fprintf('      MATLAB file: %s\n',actualFile);
            fprintf('      Expected:    %s\n',expectedFile);
            fprintf('      Found nargin/nargout: %d/%d; expected %d/%d\n', ...
                report.nargin(k),report.nargout(k), ...
                expectedInputs(k),expectedOutputs(k));
            allOK = false;
        end
    end

    testDriverPath = which('test_channel_estimator');
    expectedTestDriver = fullfile(person2Dir,'test_channel_estimator.m');
    report.test_driver_path = testDriverPath;
    report.test_driver_ok = ~isempty(testDriverPath) && ...
        strcmpi(strrep(testDriverPath,'/','\'), ...
        strrep(expectedTestDriver,'/','\'));
    if report.test_driver_ok
        fprintf('PASS  %-42s script path resolved\n','test_channel_estimator');
    else
        fprintf('FAIL  test_channel_estimator path\n');
        fprintf('      MATLAB file: %s\n',testDriverPath);
        fprintf('      Expected:    %s\n',expectedTestDriver);
        allOK = false;
    end

    report.all_ok = allOK;
    if allOK
        fprintf('\nAll Revision 20 function paths and signatures are correct.\n');
    else
        fprintf('\nInstallation check failed. Remove duplicate/stale .m files or correct the folder layout.\n');
    end
end
