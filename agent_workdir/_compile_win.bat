@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set TORCH_CUDA_ARCH_LIST=8.6
for /f "delims=" %%i in ('python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))"') do set TORCH_LIB=%%i
if defined TORCH_LIB set "PATH=%TORCH_LIB%;%PATH%"
if defined TORCH_LIB set "LIB=%TORCH_LIB%;%LIB%"
if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"
if defined CUDA_PATH set "LIB=%CUDA_PATH%\lib\x64;%LIB%"
python -m utils.compile
if exist build\forced_compile\cuda_extension.pyd (
  copy /Y build\forced_compile\cuda_extension.pyd cuda_extension.pyd >nul
  echo Copied cuda_extension.pyd
)
if exist cuda_extension.pyd (
  echo Build artifact present
  exit /b 0
)
exit /b 1
