@echo off

set BASE_DIR=%~dp0

mkdir %BASE_DIR%_data

@REM Download A2F models
set A2F_MODEL_DIR=%BASE_DIR%_data\audio2face-models
mkdir "%A2F_MODEL_DIR%"
hf download nvidia/Audio2Face-3D-v3.0          --local-dir %A2F_MODEL_DIR%/audio2face-3d-v3.0
hf download nvidia/Audio2Face-3D-v2.3.1-Claire --local-dir %A2F_MODEL_DIR%/audio2face-3d-v2.3.1-claire
hf download nvidia/Audio2Face-3D-v2.3.1-James  --local-dir %A2F_MODEL_DIR%/audio2face-3d-v2.3.1-james
hf download nvidia/Audio2Face-3D-v2.3-Mark     --local-dir %A2F_MODEL_DIR%/audio2face-3d-v2.3-mark

echo Models are downloaded to %A2F_MODEL_DIR%
