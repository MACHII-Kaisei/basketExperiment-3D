function success = detect_aruco_pose(imageFile, paramFile, outputPoseFile)
% DETECT_ARUCO_POSE 静止画からArUcoマーカーを検出し、座標変換行列を保存する
%
%   Args:
%       imageFile: ArUcoが写っている画像ファイルのパス (.jpg/.png)
%       paramFile: ステレオパラメータファイル (.mat)
%       outputPoseFile: 結果を保存するファイルパス (.mat)

    success = false;
    
    % === 設定 ===
    markerFamily = "DICT_4X4_50";
    markerID     = 0;
    markerSize   = 550; % mm
    % ============

    try
        if ~exist(paramFile, 'file')
            error('パラメータファイルが見つかりません: %s', paramFile);
        end
        
        % パラメータ読み込み
        loadedData = load(paramFile);
        if isfield(loadedData, 'calibrationSession')
            stereoParams = loadedData.calibrationSession.CameraParameters;
        elseif isfield(loadedData, 'stereoParams')
            stereoParams = loadedData.stereoParams;
        else
            error('stereoParams.mat に有効なステレオパラメータが見つかりません');
        end

        % 画像読み込み
        if ~exist(imageFile, 'file')
            error('画像ファイルが見つかりません: %s', imageFile);
        end
        I = imread(imageFile);

        % ArUco検出
        disp('MATLAB: ArUcoマーカー検出を開始します...');
        [ids, ~, poses] = readArucoMarker(I, markerFamily, stereoParams.CameraParameters1.Intrinsics, markerSize);
        
        tformCamToMarker = [];
        if ~isempty(ids)
            idx = find(ids == markerID, 1);
            if ~isempty(idx)
                tformCamToMarker = poses(idx);
                
                % 結果を保存
                save(outputPoseFile, 'tformCamToMarker');
                disp(['MATLAB: 座標変換行列を保存しました: ', outputPoseFile]);
                success = true;
            else
                disp('MATLAB: エラー - 指定されたIDのマーカーが見つかりません。');
            end
        else
            disp('MATLAB: エラー - ArUcoマーカーが検出されませんでした。');
        end

    catch ME
        disp(['MATLAB Error: ', ME.message]);
        success = false;
    end
end