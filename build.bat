@echo off
REM HPGL Build Script
REM Rebuilds the hpgl-bsd project (main DLL) in Release x64 configuration

setlocal enabledelayedexpansion

REM Set environment variables
set "MKL_ROOT=C:\Program Files (x86)\Intel\oneAPI\mkl\latest"
set "VCTargetsPath=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\"
set "SolutionDir=%~dp0src\msvc\"
set "LogFile=%~dp0build.log"

echo ========================================
echo HPGL Build Script
echo ========================================
echo.
echo MKL_ROOT: %MKL_ROOT%
echo VCTargetsPath: %VCTargetsPath%
echo Solution Dir: %SolutionDir%
echo Project: hpgl.vcxproj
echo Configuration: Release x64
echo.

REM Find BuildTools MSBuild
set "MSBUILD_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\amd64\MSBuild.exe"
if not exist "%MSBUILD_PATH%" (
    set "MSBUILD_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
)

if not exist "%MSBUILD_PATH%" (
    echo ERROR: MSBuild not found!
    pause
    exit /b 1
)

echo Using MSBuild: %MSBUILD_PATH%
echo Building...
echo.

REM Build with MSBuild (using x64 version for better performance)
"%MSBUILD_PATH%" "%SolutionDir%hpgl.vcxproj" /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=v143 /t:Rebuild /v:minimal /fl /flp:"LogFile=%LogFile%" /nologo

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Build completed successfully!
    echo ========================================
    echo.

    REM Copy built DLLs to runtime location (src\geo_bsd\) where Python loads them
    if exist "%~dp0src\msvc\geo_bsd\hpgl.dll" (
        copy /Y "%~dp0src\msvc\geo_bsd\hpgl.dll" "%~dp0src\geo_bsd\hpgl.dll" >nul
        echo   Copied hpgl.dll to src\geo_bsd\
    )
    if exist "%~dp0src\msvc\geo_bsd\_cvariogram.dll" (
        copy /Y "%~dp0src\msvc\geo_bsd\_cvariogram.dll" "%~dp0src\geo_bsd\_cvariogram.dll" >nul
        echo   Copied _cvariogram.dll to src\geo_bsd\
    )

    echo.
    echo Built files:
    if exist "%~dp0src\geo_bsd\hpgl.dll" (
        echo   - src\geo_bsd\hpgl.dll
    )
    if exist "%~dp0src\geo_bsd\_cvariogram.dll" (
        echo   - src\geo_bsd\_cvariogram.dll
    )
    echo.
    echo Build log: %LogFile%
) else (
    echo.
    echo ========================================
    echo Build FAILED!
    echo ========================================
    echo.
    echo Check %LogFile% for details.
    echo.
    type "%LogFile%"
    pause
    exit /b 1
)

echo.
endlocal
