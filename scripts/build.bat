@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

if not exist build (
    cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release || exit /b 1
)

cmake --build build --config Release || exit /b 1
ctest --test-dir build --output-on-failure || exit /b 1

