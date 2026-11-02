# Environment Documentation (docs/ENV.md)

## Hardware
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **Compute Capability**: 8.9 (Ada Lovelace, sm_89)
- **SM Count**: 24
- **VRAM**: 8188 MiB (8 GB)
- **Driver Version**: 616.64 (Driver Model: WDDM)

## Software & Toolchain
- **Operating System**: Windows 11
- **Power Configuration**: High Performance Plan (`8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`), AC Power Online (`PowerOnline=True`).
  - *Warning*: On battery power, Windows/WDDM enforces an aggressive `SW_POWER_CAP` (24W limit vs. 80W default TDP) and cuts memory clock from 8,001 MHz to 5,920 MHz, causing ~50% throughput drops. All official benchmarks require AC power connection.
- **GPU Power Limit**: 80.0 W default (Min: 20.0 W, Max: 140.0 W)
- **CUDA Toolkit**: 12.4 (`nvcc` V12.4.99)
- **Host Compiler**: Microsoft Visual Studio 2022 Community (v17.14.23, MSVC `cl.exe` 19.44.35222)
  - Note: MSVC 19.44 was accepted natively by nvcc 12.4 in CMake without requiring `-allow-unsupported-compiler`.
- **CMake**: 3.31.6 (`C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`)
- **Ninja**: 1.12.1 (`C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe`)
- **Python**: 3.10.0
- **PyTorch**: 2.6.0+cu124
- **Transformers**: 5.15.0
- **Triton**: 3.8.0
- **llama.cpp**: Commit `3c809f4` (built with CUDA backend sm_89 via CMake Ninja)

## Reference Engine Submodule
- **Repository**: `reference/py-engine` (https://github.com/XVX-016/vLLM-inference-engine)
- **Pinned Commit SHA**: `bdede99aad785b26608181301843d2a4e3226023` (Tag `v1.0` does not exist on remote; human must tag `v1.0` on upstream repo)

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

## Model Weights & GGUF Canonical Checksums
- `data/models/qwen2.5-0.5b-instruct-fp16.gguf`:
  - Size: 994,157,152 bytes (948.1021 MiB)
  - SHA-256: `0b6994e7f4d8a79731690ec2cfc88e468e1489e0a452bb71f5aa0ca369e17bf1`
  - Converted directly from HF safetensors (`convert_hf_to_gguf.py`). Tied embeddings, 290 tensors.
- `data/models/qwen2.5-0.5b-instruct-q8_0.gguf`:
  - Size: 531,068,512 bytes (506.4664 MiB)
  - SHA-256: `a67dcbddd648ed96333c8383a032996a242038bf7afbbe6ea313901159f65c22`
  - Quantized from clean FP16 via `llama-quantize.exe`. Tied embeddings, 290 tensors.
- `data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`:
  - Size: 491,400,032 bytes (468.6356 MiB)
  - SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
  - Downloaded official Qwen GGUF. Untied embeddings, 291 tensors.

## Profiler & Tracing Guidelines
- **Nsight Systems CLI (`nsys`)**:
  - The stock Nsight Systems (v2023.4.4 bundled with CUDA 12.4) fails on Windows 11 with driver 616.x when unprivileged due to `ReflexStatsTraceLoggingProvider` requiring elevation.
  - Command to run in an elevated (Administrator) terminal:
    ```cmd
    "C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.4.4\target-windows-x64\nsys.exe" profile -t cuda,nvtx -o docs/profiles/P3_nsys --force-overwrite true build\generate.exe --profile-step
    ```
- **In-Engine Profiling Fallback**:
  - Inside our C++ inference engine, per-kernel timing is achieved directly via `cudaEventCreate` / `cudaEventRecord` / `cudaEventElapsedTime` paired with high-resolution host timestamps (`std::chrono::steady_clock`), or the CUPTI activity API (`cuptiActivityEnable`, headers in `CUPTI/include`, lib `cupti.lib`).

