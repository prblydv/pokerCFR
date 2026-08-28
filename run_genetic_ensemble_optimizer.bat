@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found on PATH.
    pause
    exit /b 1
)

echo Starting resumable weighted-ensemble genetic optimization.
echo Progress, EV decomposition, fitness penalties, and ETA will appear here.
echo Results: artifacts\downloaded_risk_aware\genetic_ensemble_search_v1
echo.

python -u "%~dp0optimize_heads_up_ensemble_genetic.py" %*
if errorlevel 1 (
    echo.
    echo The genetic ensemble optimizer exited with an error.
    echo Re-run this same BAT/command to resume after correcting the error.
    pause
    exit /b 1
)

echo.
echo Optimization completed successfully.
pause
endlocal
