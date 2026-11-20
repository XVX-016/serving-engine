@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
nvcc -O3 -arch=sm_89 -I. tests\test_stripped_gemv.cu -o build\test_stripped_gemv.exe || exit /b 1
build\test_stripped_gemv.exe

