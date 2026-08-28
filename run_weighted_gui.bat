@echo off
cd /d "%~dp0"

python heads_up_gui_weighted.py ^
  --policy artifacts/heads_up_v4_paper3x/snapshots/policy_00000725.pt ^
  --policy-secondary artifacts/heads_up_v4_paper3x/snapshots/policy_00001025.pt ^
  --policy-secondary artifacts/downloaded_risk_aware/policy_00000275.pt ^
  --policy-secondary artifacts/downloaded_risk_aware/policy_00000300.pt ^
  --policy-secondary artifacts/downloaded_risk_aware/policy_00000400.pt ^
  --policy-weights 35 35 1 1 28 ^
  --no-search ^
  --top-policy-actions 2 ^
  --unmodified-policy-sampling

if errorlevel 1 (
  echo.
  echo The GUI failed to start. Review the error above.
  pause
)
