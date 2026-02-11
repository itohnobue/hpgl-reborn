@echo off
setlocal

set "VCTargetsPath=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\"
set "SolutionDir=C:\Users\itohnobue\Git\hpgl\src\"
set "MKL_ROOT=C:\Program Files (x86)\Intel\oneAPI\mkl\latest"

"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe" "C:\Users\itohnobue\Git\hpgl\src\msvc\hpgl-gpl.vcxproj" /p:Configuration=Release /p:Platform=x64 /v:minimal

endlocal
