@echo off
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "HU_NATIVE_TEMP=%TEMP%\heads_up_native_engine_build"
python setup.py build_ext --inplace --build-temp "%HU_NATIVE_TEMP%"
if errorlevel 1 exit /b 1
copy /Y heads_up_native_engine*.pyd "..\" >nul
endlocal
