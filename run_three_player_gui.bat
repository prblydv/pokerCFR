@echo off
setlocal
cd /d "%~dp0"

set "POKER_SCRIPT=%~dp0play_three_player_gui.py"
set "POKER_POLICY=%~dp0artifacts\downloaded_blueprints\policy_00008100.pt"

if not exist "%POKER_POLICY%" (
    echo ERROR: Downloaded Vast.ai policy was not found:
    echo %POKER_POLICY%
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found on PATH.
    echo Install Python and run: python -m pip install -r requirements-three-player.txt
    pause
    exit /b 1
)

echo Loading trained tournament policy with 7-second deep search:
echo %POKER_POLICY%
echo.
python "%POKER_SCRIPT%" --checkpoint "%POKER_POLICY%" --search-ms 7000 --search-rollouts 150000 --bot-delay 150 %*
if errorlevel 1 (
    echo.
    echo The poker GUI exited with an error.
    pause
    exit /b 1
)

endlocal
