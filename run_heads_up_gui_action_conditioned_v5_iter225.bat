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
    python "%~dp0play_heads_up_gui.py" --policy "%~dp0artifacts\downloaded_policies\pokerCFR_HU_action_conditioned_v5_hidden512_no_range_iter00000225_policy_only.pt" --policy-device auto --policy-mode sample --no-search --unmodified-policy-sampling --root-min-strategy-probability 0 --results-log "%~dp0artifacts\heads_up_gui_results\action_conditioned_v5_hidden512_iter225_rejected_campaign\hands.jsonl"
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
