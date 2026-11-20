@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUPTI_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\CUPTI\include
set CUPTI_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\CUPTI\lib64
set CUDA_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include
set CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64
cd /d c:\Computing\serving_engine
cl /nologo /std:c++17 /EHsc /I"%CUPTI_INC%" /I"%CUDA_INC%" tests\cupti_gap_harness.cpp /link /LIBPATH:"%CUPTI_LIB%" /LIBPATH:"%CUDA_LIB%" cupti.lib cudart.lib /OUT:build\cupti_gap_harness.exe 2>&1
echo Exit code: %ERRORLEVEL%
