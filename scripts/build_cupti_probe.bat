@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUPTI_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\CUPTI\include
set CUPTI_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\CUPTI\lib64
set CUDA_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include
cd /d c:\Computing\serving_engine
cl /nologo /I"%CUPTI_INC%" /I"%CUDA_INC%" tests\cupti_probe.cpp /link /LIBPATH:"%CUPTI_LIB%" cupti.lib /OUT:build\cupti_probe.exe 2>&1
echo Exit code: %ERRORLEVEL%
