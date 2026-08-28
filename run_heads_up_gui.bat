@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found on PATH.
    pause
    exit /b 1
)

if "%~1"=="" (
    python "%~dp0play_heads_up_gui.py" --policy "%~dp0artifacts\heads_up_v4_paper3x\snapshots\policy_00000725.pt" --policy-secondary "%~dp0artifacts\heads_up_v4_paper3x\snapshots\policy_00000950.pt" --policy-secondary "%~dp0artifacts\heads_up_v4_paper3x\snapshots\policy_00001025.pt" --policy-device auto --policy-mode sample --no-search --unmodified-policy-sampling --top-policy-actions 3 --root-min-strategy-probability 0 --results-log "%~dp0artifacts\heads_up_gui_results\average_policy_725_950_1025_top3_campaign\hands.jsonl"
) else (
    python "%~dp0play_heads_up_gui.py" %*
)
if errorlevel 1 (
    echo.
    echo The heads-up GUI exited with an error.
    pause
    exit /b 1
)

endlocal
