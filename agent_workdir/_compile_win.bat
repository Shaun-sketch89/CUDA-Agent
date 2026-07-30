@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set TORCH_CUDA_ARCH_LIST=8.6
python -m utils.compile
if exist build\forced_compile\cuda_extension.pyd (
  copy /Y build\forced_compile\cuda_extension.pyd cuda_extension.pyd >nul
  echo Copied cuda_extension.pyd
)
exit /b %ERRORLEVEL%
