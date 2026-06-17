@echo off
REM HPGL Build Script
REM Rebuilds the hpgl-bsd project.
REM Default: Release x64. Override with --config Debug|Release --platform Win32|x64

setlocal enabledelayedexpansion

REM CLI argument parsing
set "BUILD_CONFIG=Release"
set "BUILD_PLATFORM=x64"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--config" (
    set "BUILD_CONFIG=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--platform" (
    set "BUILD_PLATFORM=%~2"
    shift
    shift
    goto :parse_args
)
echo WARNING: Unknown argument: %~1
shift
goto :parse_args
:args_done

REM Validate --config
if /i not "%BUILD_CONFIG%"=="Debug" (
    if /i not "%BUILD_CONFIG%"=="Release" (
        echo ERROR: --config must be Debug or Release, got "%BUILD_CONFIG%"
        exit /b 1
    )
)
REM Validate --platform
if /i not "%BUILD_PLATFORM%"=="Win32" (
    if /i not "%BUILD_PLATFORM%"=="x64" (
        echo ERROR: --platform must be Win32 or x64, got "%BUILD_PLATFORM%"
        exit /b 1
    )
)

REM Set environment variables
REM Override these with environment variables before running build.bat if needed:
REM   set MKL_ROOT=...      (default: C:\Program Files (x86)\Intel\oneAPI\mkl\latest)
REM   set VCTargetsPath=... (default: VS 2022 BuildTools v170)
REM   set MSBUILD_PATH=...  (default: auto-detected under VS 2022 BuildTools)
if not defined MKL_ROOT set "MKL_ROOT=C:\Program Files (x86)\Intel\oneAPI\mkl\latest"
if not exist "%MKL_ROOT%\." (
    echo ERROR: MKL not found at %MKL_ROOT%
    exit /b 1
)
if not defined VCTargetsPath (
    set "VCTargetsPath=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\"
    if not exist "!VCTargetsPath!" set "VCTargetsPath=C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\"
    if not exist "!VCTargetsPath!" set "VCTargetsPath=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\"
    if not exist "!VCTargetsPath!" set "VCTargetsPath=C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Microsoft\VC\v170\"
)
if not exist "!VCTargetsPath!\" (
    echo ERROR: VCTargetsPath not found at !VCTargetsPath!
    echo Install Visual Studio 2022 with MSVC v143 toolset.
    exit /b 1
)
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
echo Configuration: %BUILD_CONFIG% %BUILD_PLATFORM%
echo.

REM Find MSBuild (can be overridden via MSBUILD_PATH env var)
REM Tries BuildTools, then Community, then Professional editions
if not defined MSBUILD_PATH (
    set "MSBUILD_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\amd64\MSBuild.exe"
    if not exist "!MSBUILD_PATH!" set "MSBUILD_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
    if not exist "!MSBUILD_PATH!" set "MSBUILD_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe"
    if not exist "!MSBUILD_PATH!" set "MSBUILD_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
    if not exist "!MSBUILD_PATH!" set "MSBUILD_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe"
    if not exist "!MSBUILD_PATH!" set "MSBUILD_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\amd64\MSBuild.exe"
    if not exist "!MSBUILD_PATH!" set "MSBUILD_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
    if not exist "!MSBUILD_PATH!" set "MSBUILD_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe"
)

if not exist "%MSBUILD_PATH%" (
    echo ERROR: MSBuild not found!
    if not defined CI pause
    endlocal
    exit /b 1
)

echo Using MSBuild: %MSBUILD_PATH%

REM Auto-detect PlatformToolset from VCTargetsPath
set "PLATFORM_TOOLSET=v143"
echo !VCTargetsPath! | find /i "v170" >nul && set "PLATFORM_TOOLSET=v143"
echo !VCTargetsPath! | find /i "v160" >nul && set "PLATFORM_TOOLSET=v142"
echo !VCTargetsPath! | find /i "v150" >nul && set "PLATFORM_TOOLSET=v141"

echo Building...
echo.

REM Build with MSBuild (using x64 version for better performance)
echo Building hpgl.vcxproj...
"%MSBUILD_PATH%" "%SolutionDir%hpgl.vcxproj" /p:Configuration=%BUILD_CONFIG% /p:Platform=%BUILD_PLATFORM% /p:PlatformToolset=%PLATFORM_TOOLSET% /t:Rebuild /v:minimal /fl /flp:"LogFile=%LogFile%" /nologo
if %ERRORLEVEL% NEQ 0 goto :build_failed

echo.
echo Building cvariogram.vcxproj...
"%MSBUILD_PATH%" "%SolutionDir%cvariogram.vcxproj" /p:Configuration=%BUILD_CONFIG% /p:Platform=%BUILD_PLATFORM% /p:PlatformToolset=%PLATFORM_TOOLSET% /t:Rebuild /v:minimal /fl /flp:"LogFile=%LogFile%;Append" /nologo
if %ERRORLEVEL% NEQ 0 goto :build_failed

REM Both builds succeeded — jump to success handler
goto :build_succeeded

:build_failed
echo.
echo ========================================
echo Build FAILED!
echo ========================================
echo.
echo Check %LogFile% for details.
echo.
type "%LogFile%"
if not defined CI pause
endlocal
exit /b 1

:build_succeeded
echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.

REM Copy built DLLs to runtime location (src\geo_bsd\) where Python loads them
REM DLL filenames differ by configuration: Release uses hpgl.dll, Debug uses hpgl_d.dll
set "DLL_SUFFIX="
if /i "%BUILD_CONFIG%"=="Debug" set "DLL_SUFFIX=_d"

REM Warn that Debug build may overwrite a Release DLL at the runtime location
if /i "%BUILD_CONFIG%"=="Debug" (
    echo WARNING: Debug build copies %DLL_SUFFIX%.dll over the runtime hpgl.dll
    echo   If a Release build was previously copied, it will be overwritten.
)

set "DLL_COPY_FAILED=0"
if exist "%~dp0src\msvc\geo_bsd\hpgl%DLL_SUFFIX%.dll" (
	copy /Y "%~dp0src\msvc\geo_bsd\hpgl%DLL_SUFFIX%.dll" "%~dp0src\geo_bsd\hpgl.dll" >nul 2>&1
	if !ERRORLEVEL! EQU 0 (
		echo   Copied hpgl%DLL_SUFFIX%.dll to src\geo_bsd\hpgl.dll
	) else (
		echo   ERROR: Failed to copy hpgl%DLL_SUFFIX%.dll
		set "DLL_COPY_FAILED=1"
	)
) else (
	echo   ERROR: hpgl%DLL_SUFFIX%.dll not found at src\msvc\geo_bsd\
	set "DLL_COPY_FAILED=1"
)
if exist "%~dp0src\msvc\geo_bsd\_cvariogram%DLL_SUFFIX%.dll" (
	copy /Y "%~dp0src\msvc\geo_bsd\_cvariogram%DLL_SUFFIX%.dll" "%~dp0src\geo_bsd\_cvariogram.dll" >nul 2>&1
	if !ERRORLEVEL! EQU 0 (
		echo   Copied _cvariogram%DLL_SUFFIX%.dll to src\geo_bsd\_cvariogram.dll
	) else (
		echo   ERROR: Failed to copy _cvariogram%DLL_SUFFIX%.dll
		set "DLL_COPY_FAILED=1"
	)
) else (
	echo   ERROR: _cvariogram%DLL_SUFFIX%.dll not found at src\msvc\geo_bsd\
	set "DLL_COPY_FAILED=1"
)

if "!DLL_COPY_FAILED!"=="1" (
	echo.
	echo ERROR: DLL copy step failed.
	if not defined CI pause
	endlocal
	exit /b 1
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
endlocal
exit /b 0
