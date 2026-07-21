# Build the live-captions Windows installer end to end.
#
#   .\packaging\build.ps1              # PyInstaller bundle + Inno installer
#   .\packaging\build.ps1 -SkipInstaller   # just the PyInstaller bundle
#
# Prereqs (in the project venv):  pip install -e ".[all,dev]"
#   plus:  Inno Setup 6 (iscc.exe on PATH), and packaging\vc_redist.x64.exe present.
# Run from the repo root with the venv active.

param([switch]$SkipInstaller)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "== 1/3  Cleaning previous build ==" -ForegroundColor Cyan
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

Write-Host "== 2/3  PyInstaller (Tier A --onedir; this takes a while) ==" -ForegroundColor Cyan
pyinstaller packaging\livecaptions.spec --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# Surface anything heavy that got EXCLUDED by mistake, or MISSING dynamic imports.
$warn = Get-ChildItem build\*\warn-*.txt -ErrorAction SilentlyContinue | Select-Object -First 1
if ($warn) {
    Write-Host "  Reviewing $($warn.Name) for missing ML imports..." -ForegroundColor DarkGray
    $hits = Select-String -Path $warn.FullName -Pattern "missing module named (nemo|pyannote|torch|lightning)" -ErrorAction SilentlyContinue
    if ($hits) {
        Write-Host "  WARNING: possible missing dynamic imports (add to the spec's hiddenimports):" -ForegroundColor Yellow
        $hits | Select-Object -First 15 | ForEach-Object { Write-Host "    $($_.Line)" -ForegroundColor Yellow }
    }
}

# Confirm the load-bearing assets actually landed in the bundle.
$internal = "dist\LiveCaptions\_internal"
$checks = @(
    @{ Path = "dist\LiveCaptions\livecaptions.exe";          What = "CLI exe" },
    @{ Path = "dist\LiveCaptions\livecaptions-overlay.exe";  What = "overlay exe" },
    @{ Path = "$internal\nvidia\cublas\bin\cublas64_12.dll"; What = "cuBLAS (GPU Whisper)" },
    @{ Path = "$internal\ctranslate2\ctranslate2.dll";       What = "CTranslate2" }
)
$vad = Get-ChildItem "$internal\faster_whisper" -Recurse -Filter "silero_vad*.onnx" -ErrorAction SilentlyContinue
foreach ($c in $checks) {
    if (Test-Path $c.Path) { Write-Host "  ok  $($c.What)" -ForegroundColor Green }
    else { Write-Host "  MISSING  $($c.What)  ($($c.Path))" -ForegroundColor Red }
}
if ($vad) { Write-Host "  ok  Silero VAD model" -ForegroundColor Green }
else { Write-Host "  MISSING  Silero VAD model (transcribe will FileNotFoundError)" -ForegroundColor Red }

$size = (Get-ChildItem dist\LiveCaptions -Recurse | Measure-Object Length -Sum).Sum / 1GB
Write-Host ("  bundle size: {0:N2} GB" -f $size) -ForegroundColor DarkGray

if ($SkipInstaller) { Write-Host "Done (bundle only)." -ForegroundColor Cyan; exit 0 }

Write-Host "== 3/3  Inno Setup installer ==" -ForegroundColor Cyan
if (-not (Test-Path "packaging\vc_redist.x64.exe")) {
    throw "packaging\vc_redist.x64.exe missing — download it from aka.ms/vs/17/release/vc_redist.x64.exe"
}
$iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
if (-not $iscc) { throw "iscc.exe not on PATH — install Inno Setup 6.3 or newer (x64os needs 6.3+)" }
# The build version, read from the package, drives the patch payload + manifest.
$ver = (Select-String -Path "src\livecaptions\__init__.py" -Pattern '__version__\s*=\s*"([^"]+)"').Matches.Groups[1].Value

# Differential update prep: hash _internal (into the bundle), stage the patch payload
# (just the exes + that hash), and write Output\manifest.json. Must run BEFORE the
# installers compile so internal.sha256 ships inside BOTH of them.
Write-Host "== 3/4  Differential update payload ==" -ForegroundColor Cyan
$staging = "dist\LiveCaptions-patch"
python packaging\make_patch.py "dist\LiveCaptions" $staging "Output" $ver
if ($LASTEXITCODE -ne 0) { throw "make_patch.py failed" }

Write-Host "== 4/4  Inno Setup installers (full + patch) ==" -ForegroundColor Cyan
& $iscc.Source packaging\livecaptions.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed (full)" }
# Same .iss, PatchBuild define + the staged SourceDir: a small in-place upgrade
# that carries only the exes. Only offered by the updater when the _internal hash
# proves the libraries match; a fresh install always uses the full installer.
& $iscc.Source "/DPatchBuild" "/DSourceDir=..\dist\LiveCaptions-patch" packaging\livecaptions.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed (patch)" }

# OutputDir=..\Output in the .iss lands the installers at repo\Output.
foreach ($pat in @("LiveCaptions-Setup-*.exe", "LiveCaptions-Patch-*.exe")) {
    $f = Get-ChildItem "Output\$pat" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($f) { Write-Host ("  {0}  ({1:N0} MB)" -f $f.Name, ($f.Length/1MB)) -ForegroundColor Green }
}
Write-Host "Done." -ForegroundColor Cyan
