function run_tracking_func(videoFileL, videoFileR, csvOutputName, paramFile, poseFile)
% RUN_TRACKING_FUNC (YOLO版 + 正しいサイズでの描画修正版)
%   YOLOv8/v11を使用してボール検出を行い、3次元計測を行います。
%   検出されたバウンディングボックスのサイズを正しく反映して描画します。

    % === 設定エリア ===
    outputDebugVideo = true; 
    modelPath = 'models/best-yolo11n.pt'; % ユーザー指定のファイル名
    confidenceThreshold = 0.4;
    windowSize = 5;
    % =================

    try
        %% --- Step 1: 準備 & GPUチェック ---
        if ~exist(paramFile, 'file'), error('パラメータファイルなし: %s', paramFile); end
        
        loadedData = load(paramFile);
        if isfield(loadedData, 'calibrationSession')
            stereoParams = loadedData.calibrationSession.CameraParameters;
        elseif isfield(loadedData, 'stereoParams')
            stereoParams = loadedData.stereoParams;
        else
            error('stereoParamsが無効です');
        end
        
        loadedPose = load(poseFile);
        if isfield(loadedPose, 'tformCamToMarker')
            tformCamToMarker = loadedPose.tformCamToMarker;
        else
            error('poseFileが無効です');
        end

        % GPUチェック
        device = 'cpu'; 
        try
            isGPU = py.torch.cuda.is_available();
            if isGPU
                device = '0'; 
                gpuName = string(py.torch.cuda.get_device_name(0));
                disp('================================================');
                disp(['MATLAB: ★ GPUが有効です! 使用デバイス: ', char(gpuName)]);
                disp('================================================');
            else
                disp('MATLAB: ⚠ GPUが見つかりません。CPUモードで実行します。');
            end
        catch
            disp('MATLAB: PyTorchの確認に失敗しました。CPUを使用します。');
        end

        %% --- モデルロード ---
        disp('MATLAB: YOLOモデルをロード中...');
        try
            if count(py.sys.path, pwd) == 0
                insert(py.sys.path, int32(0), pwd);
            end

            loader_mod = py.importlib.import_module('yolo_loader');
            py.importlib.reload(loader_mod);
            
            yolo_model = loader_mod.get_model(modelPath);
            disp('MATLAB: モデルロード成功');
            
        catch ME
            error('YOLOモデルのロードに失敗しました。\nヒント: yolo_loader.py が同じフォルダにあるか確認してください。\nエラー詳細: %s', ME.message);
        end

        vL = VideoReader(videoFileL);
        vR = VideoReader(videoFileR);

        % デバッグ動画の準備
        if outputDebugVideo
            [folder, name, ~] = fileparts(csvOutputName);
            debugFileL = fullfile(folder, [name, '_debug_left.mp4']);
            debugFileR = fullfile(folder, [name, '_debug_right.mp4']);
            
            vWriterL = VideoWriter(debugFileL, 'MPEG-4');
            vWriterR = VideoWriter(debugFileR, 'MPEG-4');
            vWriterL.FrameRate = vL.FrameRate;
            vWriterR.FrameRate = vR.FrameRate;
            open(vWriterL);
            open(vWriterR);
        end

        %% --- Step 2: YOLO検出フェーズ ---
        disp('MATLAB: YOLOによる検出を実行中...');
        
        coordL = []; 
        coordR = []; 
        frameCount = 0;
        
        tempImgL = fullfile(pwd, 'temp_yolo_L.jpg');
        tempImgR = fullfile(pwd, 'temp_yolo_R.jpg');
        
        while hasFrame(vL) && hasFrame(vR)
            frameCount = frameCount + 1;
            
            frameL = readFrame(vL);
            frameR = readFrame(vR);
            
            % 【修正】bbox情報(x,y,w,h)を全て受け取る
            detL = detectWithYOLO(frameL, yolo_model, loader_mod, tempImgL, confidenceThreshold, device);
            detR = detectWithYOLO(frameR, yolo_model, loader_mod, tempImgR, confidenceThreshold, device);
            
            % 座標保存 (中心座標 x, y だけを取り出して保存)
            if ~isempty(detL)
                centerL = detL(1:2);
                coordL = [coordL; frameCount, centerL]; 
            end
            if ~isempty(detR)
                centerR = detR(1:2);
                coordR = [coordR; frameCount, centerR]; 
            end
            
            % デバッグ書き込み
            if outputDebugVideo
                imgOutL = frameL;
                if ~isempty(detL)
                    centerL = detL(1:2);
                    w = detL(3); h = detL(4);
                    % 【修正】検出された通りのサイズで矩形を作る [左上x, 左上y, 幅, 高さ]
                    bbox = [centerL(1)-w/2, centerL(2)-h/2, w, h];
                    
                    imgOutL = insertMarker(imgOutL, centerL, 'x', 'Color', 'green', 'Size', 15);
                    imgOutL = insertShape(imgOutL, 'Rectangle', bbox, 'Color', 'green', 'LineWidth', 2);
                end
                writeVideo(vWriterL, imgOutL);
                
                imgOutR = frameR;
                if ~isempty(detR)
                    centerR = detR(1:2);
                    w = detR(3); h = detR(4);
                    bbox = [centerR(1)-w/2, centerR(2)-h/2, w, h];
                    
                    imgOutR = insertMarker(imgOutR, centerR, 'x', 'Color', 'green', 'Size', 15);
                    imgOutR = insertShape(imgOutR, 'Rectangle', bbox, 'Color', 'green', 'LineWidth', 2);
                end
                writeVideo(vWriterR, imgOutR);
            end

            if mod(frameCount, 50) == 0 
                disp(['MATLAB: Analyzing Frame ', num2str(frameCount)]);
            end
        end
        
        if exist(tempImgL, 'file'), delete(tempImgL); end
        if exist(tempImgR, 'file'), delete(tempImgR); end

        if outputDebugVideo
            close(vWriterL);
            close(vWriterR);
        end

        %% --- Step 3: 自動同期補正 ---
        if isempty(coordL) || isempty(coordR)
            disp('MATLAB: ボールが十分に検出されませんでした。');
            return;
        end
        
        maxFrame = max(max(coordL(:,1)), max(coordR(:,1)));
        signalL = zeros(maxFrame, 1);
        signalR = zeros(maxFrame, 1);
        signalL(coordL(:,1)) = coordL(:,3);
        signalR(coordR(:,1)) = coordR(:,3);
        
        disp('MATLAB: 同期ズレを計算中...');
        [acor, lag] = xcorr(signalL, signalR);
        [~, I] = max(abs(acor));
        timeDiff = lag(I);
        
        disp(['MATLAB: 推定された同期ズレ (Right - Left): ', num2str(timeDiff), ' frames']);
        
        %% --- Step 4: 3次元復元 ---
        fullTrajectory = [];
        
        for i = 1:size(coordL, 1)
            fNumL = coordL(i, 1);
            targetFrameR = fNumL - timeDiff;
            idxR = find(coordR(:,1) == targetFrameR);
            
            if ~isempty(idxR)
                ptL = coordL(i, 2:3);
                ptR = coordR(idxR, 2:3);
                
                pL_ud = undistortPoints(ptL, stereoParams.CameraParameters1);
                pR_ud = undistortPoints(ptR, stereoParams.CameraParameters2);
                ptCam = triangulate(pL_ud, pR_ud, stereoParams);
                ptMarker = transformPointsForward(tformCamToMarker, ptCam);
                
                if abs(ptMarker(1)) < 20000 && abs(ptMarker(2)) < 20000 && abs(ptMarker(3)) < 10000
                    fullTrajectory = [fullTrajectory; ptMarker];
                end
            end
        end

        %% --- Step 5: CSV出力 ---
        if ~isempty(fullTrajectory)
            if size(fullTrajectory, 1) > windowSize
                fullTrajectory = smoothdata(fullTrajectory, 'movmean', windowSize);
            end
        
            outTable = array2table(fullTrajectory, 'VariableNames', {'X_mm', 'Y_mm', 'Z_mm'});
            writetable(outTable, csvOutputName);
            disp(['MATLAB: 書き出し完了: ', csvOutputName]);
            
            hFig = figure('Visible', 'off');
            hAxes = axes(hFig);
            grid(hAxes, 'on'); axis(hAxes, 'equal'); view(hAxes, 3);
            xlabel(hAxes, 'X [mm]'); ylabel(hAxes, 'Y [mm]'); zlabel(hAxes, 'Z [mm]');
            title(hAxes, '3D Trajectory (YOLO & GPU)');
            hold(hAxes, 'on');
            
            plot3(hAxes, fullTrajectory(:,1), fullTrajectory(:,2), fullTrajectory(:,3), 'b.-');
            [folder, name, ~] = fileparts(csvOutputName);
            saveas(hFig, fullfile(folder, [name, '_plot.png']));
            close(hFig);
        else
            disp('MATLAB: 警告: 有効な3次元軌跡が生成されませんでした');
        end
        
    catch ME
        if exist('outputDebugVideo', 'var') && outputDebugVideo
            if exist('vWriterL', 'var'), close(vWriterL); end
            if exist('vWriterR', 'var'), close(vWriterR); end
        end
        disp(['MATLAB Error: ', ME.message]);
        rethrow(ME);
    end
end

%% --- ヘルパー関数: YOLO検出 ---
function bbox = detectWithYOLO(img, model, loader, tempPath, confThresh, device)
    % 返り値を [cx, cy, w, h] の4要素に変更
    bbox = [];
    imwrite(img, tempPath);
    
    results = loader.detect_safe(model, tempPath, confThresh, device);
    
    if ~isempty(results)
        if isa(results, 'py.list')
            data_cell = cell(results);
            if ~isempty(data_cell)
                % 最初の検出結果 (x, y, w, h) をすべて取得
                bbox = double(data_cell{1}); 
            end
        end
    end
end
