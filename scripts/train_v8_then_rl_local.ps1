param(
  [string]$StartDate = "2023-01-01",
  [string]$EndDate = "2026-01-01",
  [string]$DataRoot = "dataset_cloudforce_radiation_v6",
  [string]$V8OutDir = "runs/cloud_radiation_bottom_v8_clean_direct",
  [string]$RlOutDir = "runs/cloud_rl_v8_radiation",
  [int]$MaxScenesPerLocation = 200,
  [int]$ResolutionM = 250,
  [int]$PatchKm = 10,
  [int]$S2Workers = 4,
  [int]$OpenMeteoWorkers = 1,
  [int]$V8Epochs = 120,
  [int]$V8BatchSize = 24,
  [int]$RlUpdates = 1000,
  [int]$RlBatchSize = 16,
  [switch]$SkipDataset,
  [switch]$SkipDeps,
  [switch]$FreshRl
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

if (-not $SkipDataset -and (-not $env:SH_CLIENT_ID -or -not $env:SH_CLIENT_SECRET)) {
  throw "Missing Sentinel Hub credentials. Set `$env:SH_CLIENT_ID and `$env:SH_CLIENT_SECRET before running this script."
}

if (-not $SkipDeps) {
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt -r cloud_rl/requirements.txt
}

$DataRootPath = Join-Path $RepoRoot $DataRoot
$V8OutPath = Join-Path $RepoRoot $V8OutDir
$RlOutPath = Join-Path $RepoRoot $RlOutDir

if (-not $SkipDataset) {
  python NewModel/build_temperature_dataset_cloudforce_radiation_v6.py `
    --start-date $StartDate `
    --end-date $EndDate `
    --out $DataRootPath `
    --max-scenes-per-location $MaxScenesPerLocation `
    --resolution-m $ResolutionM `
    --patch-km $PatchKm `
    --s2-workers $S2Workers `
    --openmeteo-workers $OpenMeteoWorkers `
    --target-offset-days 5
}

python NewModel/train_cloud_radiation_bottom_v8_CLEAN_DIRECT.py `
  --data-root $DataRootPath `
  --out-dir $V8OutPath `
  --image-height 160 `
  --image-width 160 `
  --lookback 4 `
  --max-gap-days 12 `
  --use-cloud-tensor `
  --epochs $V8Epochs `
  --batch-size $V8BatchSize `
  --num-workers 0 `
  --augment `
  --channels-last `
  --lr 3e-4 `
  --weight-decay 3e-4 `
  --loss-scale 300 `
  --min-clear-wm2 120 `
  --early-stop-patience 30

$resumeMode = "auto"
if ($FreshRl) {
  $resumeMode = "none"
}

python cloud_rl/train_rl.py `
  --config cloud_rl/configs/default.yaml `
  --data-root $DataRootPath `
  --out-dir $RlOutPath `
  --split train `
  --resume $resumeMode `
  --updates $RlUpdates `
  --batch-size $RlBatchSize `
  --reward-checkpoint (Join-Path $V8OutPath "best.pt") `
  --reward-scale 120 `
  --reward-improvement-gain 10 `
  --reward-absolute-error-weight 1 `
  --reward-budget-penalty 0.02

Write-Host ""
Write-Host "Done."
Write-Host "V8 reward checkpoint: $(Join-Path $V8OutPath 'best.pt')"
Write-Host "RL checkpoint: $(Join-Path $RlOutPath 'policy_latest.pt')"
