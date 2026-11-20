@echo off
cd /d c:\Computing\serving_engine
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat" --csv --metrics gpu__time_duration.sum,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,smsp__inst_executed.sum --target-processes all build\tests\bench_persistent_shape.exe 2
