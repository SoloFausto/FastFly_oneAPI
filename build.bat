@echo off
setlocal
REM Run from an Intel oneAPI command prompt with Visual Studio C++ tools.
set "BUILD_TYPE=Release"
if /I "%~1"=="clean" goto clean
if /I "%~1"=="debug" set "BUILD_TYPE=Debug"
where icx >nul 2>&1
if errorlevel 1 (
    echo ERROR: icx not found. Initialize the Intel oneAPI DPC++/C++ environment.
    exit /b 1
)
where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: CMake 3.20 or later is required.
    exit /b 1
)
where ninja >nul 2>&1
if errorlevel 1 (
    echo ERROR: Ninja is required. Install it and add it to PATH.
    exit /b 1
)
cmake -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
if errorlevel 1 exit /b 1
cmake --build "%~dp0build" --parallel
if errorlevel 1 exit /b 1
echo Build successful. Run build\flywire_sim.exe or python flywire_sim.py
exit /b 0

:clean
if not exist "%~dp0build\CMakeCache.txt" exit /b 0
cmake --build "%~dp0build" --target clean
exit /b %errorlevel%
