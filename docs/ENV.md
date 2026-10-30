# Environment Documentation (docs/ENV.md)

## Hardware
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **Compute Capability**: 8.9 (Ada Lovelace, sm_89)
- **SM Count**: 24
- **VRAM**: 8188 MiB (8 GB)
- **Driver Version**: 616.64 (Driver Model: WDDM)

## Software & Toolchain
- **Operating System**: Windows 11
- **CUDA Toolkit**: 12.4 (`nvcc` V12.4.99)
- **Host Compiler**: Microsoft Visual Studio 2022 Community (v17.14.23, MSVC `cl.exe` 19.44.35222)
  - Note: MSVC 19.44 was accepted natively by nvcc 12.4 in CMake without requiring `-allow-unsupported-compiler`.
- **CMake**: 3.31.6 (`C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`)
- **Ninja**: 1.12.1 (`C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe`)
- **Python**: 3.10.0
- **PyTorch**: 2.6.0+cu124
- **Transformers**: 5.15.0
- **Triton**: 3.8.0

## Reference Engine Submodule
- **Repository**: `reference/py-engine` (https://github.com/XVX-016/vLLM-inference-engine)
- **Pinned Commit SHA**: `bdede99aad785b26608181301843d2a4e3226023` (tag `v1.0` / main freeze)

## Build and Test Commands

### 1. Developer Environment Initialization
To configure and build with Ninja and MSVC from PowerShell or CMD, you can directly run:

```cmd
scripts\build.bat
```

Or manually initialize the x64 developer environment:
```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%" && cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release && ctest --test-dir build --output-on-failure'
```

### 2. Configure (using Ninja)
```powershell
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
```

### 3. Build
```powershell
cmake --build build --config Release
```

### 4. Run Tests
```powershell
ctest --test-dir build --output-on-failure
```
