function [success, calibInfo] = calibrate_from_backboard(pointsLeft, pointsRight, paramFile, outputPoseFile, options)
%CALIBRATE_FROM_BACKBOARD バックボード4点からカメラ→ArUco変換行列を計算
%
% 入力:
%   pointsLeft  - [4x2] 左カメラ画像上のバックボード4角座標 (左上,右上,右下,左下)
%   pointsRight - [4x2] 右カメラ画像上のバックボード4角座標 (同順序)
%   paramFile   - stereoParams.mat のパス
%   outputPoseFile - 出力先 (marker_pose.mat)
%   options     - オプション構造体 (省略可)
%       .backboard_width  = 1800 (mm)
%       .backboard_height = 1050 (mm)
%       .aruco_distance_from_backboard = 2572 (mm)
%       .ring_height = 3050 (mm)
%
% 出力:
%   success   - 成功フラグ
%   calibInfo - キャリブレーション情報構造体

    success = false;
    calibInfo = struct();
    
    %% デフォルトオプション設定
    if nargin < 5 || isempty(options)
        options = struct();
    end
    if ~isfield(options, 'backboard_width')
        options.backboard_width = 1800;  % mm
    end
    if ~isfield(options, 'backboard_height')
        options.backboard_height = 1050;  % mm
    end
    if ~isfield(options, 'aruco_distance_from_backboard')
        options.aruco_distance_from_backboard = 2600;  % mm (4600 - 2000) FIBA規格
    end
    if ~isfield(options, 'ring_height')
        options.ring_height = 3050;  % mm
    end
    if ~isfield(options, 'freethrow_distance')
        options.freethrow_distance = 4600;  % mm (FIBA国際基準)
    end
    
    try
        %% Step 1: ステレオパラメータ読み込み
        disp('=== バックボード4点キャリブレーション開始 ===');
        
        if ~exist(paramFile, 'file')
            error('ステレオパラメータファイルが見つかりません: %s', paramFile);
        end
        
        loadedData = load(paramFile);
        if isfield(loadedData, 'calibrationSession')
            stereoParams = loadedData.calibrationSession.CameraParameters;
        elseif isfield(loadedData, 'stereoParams')
            stereoParams = loadedData.stereoParams;
        else
            error('stereoParams.mat に有効なステレオパラメータが見つかりません');
        end
        disp('ステレオパラメータ読み込み完了');
        
        %% Step 2: 入力検証
        if size(pointsLeft, 1) ~= 4 || size(pointsLeft, 2) ~= 2
            error('pointsLeft は [4x2] の行列である必要があります');
        end
        if size(pointsRight, 1) ~= 4 || size(pointsRight, 2) ~= 2
            error('pointsRight は [4x2] の行列である必要があります');
        end
        
        %% Step 3: 歪み補正 & ステレオ三角測量
        disp('ステレオ三角測量を実行中...');
        
        % 歪み補正
        pointsLeft_ud = undistortPoints(pointsLeft, stereoParams.CameraParameters1);
        pointsRight_ud = undistortPoints(pointsRight, stereoParams.CameraParameters2);
        
        % 三角測量 (カメラ座標系での3D点)
        points3D_cam = triangulate(pointsLeft_ud, pointsRight_ud, stereoParams);
        
        % NaN/Infチェック
        if any(isnan(points3D_cam(:))) || any(isinf(points3D_cam(:)))
            error('三角測量に失敗しました（NaN/Inf検出）');
        end
        
        disp('三角測量結果 (カメラ座標系):');
        disp(points3D_cam);
        
        calibInfo.backboard_points_3d = points3D_cam;
        
        %% Step 4: バックボードサイズ検証
        % P1(左上), P2(右上), P3(右下), P4(左下)
        P1 = points3D_cam(1, :);
        P2 = points3D_cam(2, :);
        P3 = points3D_cam(3, :);
        P4 = points3D_cam(4, :);
        
        % 計測されたサイズ
        width_top = norm(P2 - P1);
        width_bottom = norm(P3 - P4);
        height_left = norm(P4 - P1);
        height_right = norm(P3 - P2);
        
        measured_width = (width_top + width_bottom) / 2;
        measured_height = (height_left + height_right) / 2;
        
        width_error = abs(measured_width - options.backboard_width);
        height_error = abs(measured_height - options.backboard_height);
        
        disp('--- サイズ検証 ---');
        fprintf('計測幅: %.1f mm (期待値: %.1f mm, 誤差: %.1f mm)\n', ...
            measured_width, options.backboard_width, width_error);
        fprintf('計測高さ: %.1f mm (期待値: %.1f mm, 誤差: %.1f mm)\n', ...
            measured_height, options.backboard_height, height_error);
        
        calibInfo.measured_width = measured_width;
        calibInfo.measured_height = measured_height;
        calibInfo.width_error = width_error;
        calibInfo.height_error = height_error;
        
        % 警告表示（誤差が大きい場合）
        if width_error > 100 || height_error > 100
            warning('サイズ誤差が100mmを超えています。点の指定を確認してください。');
        end
        
        %% Step 5: バックボード座標系の構築
        disp('バックボード座標系を構築中...');
        
        % バックボード座標系の定義:
        %   原点: バックボード中心
        %   X軸: 右方向 (左上→右上)
        %   Y軸: 下方向 (左上→左下)
        %   Z軸: 手前方向（カメラ側、シュート側）
        
        % 原点: 4点の重心
        origin_bb = mean(points3D_cam, 1);
        
        % X軸: 左上→右上 (正規化)
        vec_x = P2 - P1;
        vec_x = vec_x / norm(vec_x);
        
        % 仮Y軸: 左上→左下 (正規化、下方向)
        vec_y_temp = P4 - P1;
        vec_y_temp = vec_y_temp / norm(vec_y_temp);
        
        % Z軸: X × Y (右手系、手前方向)
        vec_z = cross(vec_x, vec_y_temp);
        vec_z = vec_z / norm(vec_z);
        
        % Y軸を再計算 (直交性保証)
        vec_y = cross(vec_z, vec_x);
        vec_y = vec_y / norm(vec_y);
        
        % 回転行列: カメラ座標系 → バックボード座標系
        % R の各行がバックボード座標系の軸ベクトル（カメラ座標系表現）
        R_cam_to_bb = [vec_x; vec_y; vec_z];
        
        % 平行移動: バックボード原点のカメラ座標系表現を変換
        t_cam_to_bb = -R_cam_to_bb * origin_bb';
        
        disp('バックボード座標系:');
        fprintf('  原点(カメラ座標系): [%.1f, %.1f, %.1f]\n', origin_bb(1), origin_bb(2), origin_bb(3));
        fprintf('  X軸(右): [%.3f, %.3f, %.3f]\n', vec_x(1), vec_x(2), vec_x(3));
        fprintf('  Y軸(下): [%.3f, %.3f, %.3f]\n', vec_y(1), vec_y(2), vec_y(3));
        fprintf('  Z軸(手前): [%.3f, %.3f, %.3f]\n', vec_z(1), vec_z(2), vec_z(3));
        
        % rigidtform3d オブジェクト作成
        tformCamToBackboard = rigidtform3d(R_cam_to_bb, t_cam_to_bb');
        calibInfo.tformCamToBackboard = tformCamToBackboard;
        
        %% Step 6: バックボード座標系 → ArUco座標系の変換
        disp('ArUco座標系への変換行列を計算中...');
        
        % ArUco座標系の定義（目標）:
        %   原点: ArUcoマーカー中心（床面上）
        %   X軸: コート横方向（バックボードと平行、右方向）
        %   Y軸: シュート方向（ArUco→バックボード、奥方向）
        %   Z軸: 上方向
        
        % バックボード座標系:
        %   X_bb: 右方向（左上→右上）
        %   Y_bb: 下方向（左上→左下）
        %   Z_bb: 手前方向（カメラ側、X×Yで決まる）
        
        % バックボード座標系からArUco座標系への軸対応（右手系を保つ）:
        %   X_aruco = X_bb (横方向、同じ)
        %   Y_aruco = Z_bb (シュート方向 = バックボード手前から奥へ、つまりZ_bb方向)
        %   Z_aruco = -Y_bb (上方向 = バックボード下方向の反転)
        % 
        % det([1,0,0; 0,0,1; 0,-1,0]) = 1*(0*0 - 1*(-1)) = 1 ✓
        R_bb_to_aruco = [
            1,  0,  0;   % X_aruco = X_bb
            0,  0,  1;   % Y_aruco = Z_bb
            0, -1,  0;   % Z_aruco = -Y_bb
        ];
        
        % 検証: det(R) = 1 ✓
        det_R = det(R_bb_to_aruco);
        fprintf('回転行列の行列式: %.3f (1であるべき)\n', det_R);
        
        % ArUcoの位置（バックボード座標系で表現）:
        %   X: 0 (バックボード中心線上)
        %   Y: バックボード中心から床面までの距離（下方向が正）
        %   Z: バックボード面からArUcoまでの距離（手前方向が正、つまり負の値）
        %
        % バックボード中心の高さを計算:
        %   リング高さ: 3050mm
        %   リングはバックボード下端から約150mm上にある
        %   バックボード高さ: 1050mm
        %   バックボード下端高さ = リング高さ - 150mm = 2900mm
        %   バックボード中心高さ = バックボード下端 + 高さ/2 = 2900 + 525 = 3425mm
        %
        % ただし、実際のFIBA規格では:
        %   バックボード下端からリング上端までの距離: 約290mm
        %   リング直径: 450mm → リング中心はリング上端から225mm下
        %   よって、バックボード下端からリング中心: 290 - 225 = 65mm（リングがバックボードより下）
        %   → バックボード下端高さ = 3050 + 65 = 3115mm
        %   → バックボード中心高さ = 3115 + 525 = 3640mm
        %
        % 簡略化: バックボード中心高さ ≒ リング高さ + バックボード高さ/2 - 150mm
        backboard_center_height = options.ring_height + options.backboard_height/2 - 150;
        fprintf('バックボード中心高さ: %.1f mm\n', backboard_center_height);
        
        aruco_x_bb = 0;
        aruco_y_bb = backboard_center_height;  % バックボード中心から床面までの距離
        aruco_z_bb = -options.aruco_distance_from_backboard;  % バックボード面から奥方向（負）
        
        % ArUco座標系での原点は(0,0,0)なので、バックボード座標系からの平行移動
        t_bb_to_aruco_in_bb = [aruco_x_bb; aruco_y_bb; aruco_z_bb];
        
        % 平行移動をArUco座標系に変換
        t_bb_to_aruco = -R_bb_to_aruco * t_bb_to_aruco_in_bb;
        
        tformBackboardToAruco = rigidtform3d(R_bb_to_aruco, t_bb_to_aruco');
        calibInfo.tformBackboardToAruco = tformBackboardToAruco;
        
        fprintf('ArUco位置(バックボード座標系): [%.1f, %.1f, %.1f]\n', ...
            aruco_x_bb, aruco_y_bb, aruco_z_bb);
        
        %% Step 7: 変換行列の合成
        disp('最終変換行列を合成中...');
        
        % カメラ座標系 → ArUco座標系
        % P_aruco = T_bb_to_aruco * T_cam_to_bb * P_cam
        
        % rigidtform3d の合成
        % 注: MATLAB R2022b以降では直接 * で合成可能
        % それ以前のバージョン用に手動合成も用意
        
        try
            % 新しい方法を試す
            tformCamToMarker = rigidtform3d( ...
                tformBackboardToAruco.R * tformCamToBackboard.R, ...
                (tformBackboardToAruco.R * tformCamToBackboard.Translation' + tformBackboardToAruco.Translation')');
        catch
            % 手動で合成
            R_final = R_bb_to_aruco * R_cam_to_bb;
            t_final = R_bb_to_aruco * t_cam_to_bb + t_bb_to_aruco;
            tformCamToMarker = rigidtform3d(R_final, t_final');
        end
        
        calibInfo.tformCamToMarker = tformCamToMarker;
        
        %% Step 8: 検証 - バックボード4点をArUco座標系に変換
        disp('--- 検証: バックボード4点のArUco座標系表現 ---');
        points3D_aruco = transformPointsForward(tformCamToMarker, points3D_cam);
        disp('バックボード4点 (ArUco座標系):');
        fprintf('  左上: [%.1f, %.1f, %.1f]\n', points3D_aruco(1,1), points3D_aruco(1,2), points3D_aruco(1,3));
        fprintf('  右上: [%.1f, %.1f, %.1f]\n', points3D_aruco(2,1), points3D_aruco(2,2), points3D_aruco(2,3));
        fprintf('  右下: [%.1f, %.1f, %.1f]\n', points3D_aruco(3,1), points3D_aruco(3,2), points3D_aruco(3,3));
        fprintf('  左下: [%.1f, %.1f, %.1f]\n', points3D_aruco(4,1), points3D_aruco(4,2), points3D_aruco(4,3));
        
        % Z座標（高さ）がリング高さ付近であることを確認
        avg_z = mean(points3D_aruco(:, 3));
        fprintf('  平均Z座標(高さ): %.1f mm (期待値: %.1f mm付近)\n', avg_z, options.ring_height);
        
        calibInfo.backboard_points_aruco = points3D_aruco;
        
        %% Step 9: 保存
        disp('変換行列を保存中...');
        save(outputPoseFile, 'tformCamToMarker', 'calibInfo');
        fprintf('保存完了: %s\n', outputPoseFile);
        
        success = true;
        disp('=== キャリブレーション完了 ===');
        
    catch ME
        disp(['エラー: ', ME.message]);
        calibInfo.error = ME.message;
        success = false;
    end
end
