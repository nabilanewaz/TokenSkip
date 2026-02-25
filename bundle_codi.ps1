# Bundle CODI training files
Write-Host "Bundling CODI..." -ForegroundColor Cyan
$d="codi_bundle"; New-Item -ItemType Directory -Force -Path $d | Out-Null
Copy-Item "codi_workspace\CODI\train_fixed.py" "$d\train.py"
New-Item -ItemType Directory -Force -Path "$d\src" | Out-Null
Copy-Item "codi_workspace\CODI\src\model.py" "$d\src\model.py"
New-Item -ItemType File -Force -Path "$d\src\__init__.py" | Out-Null
Write-Host "Done! Bundle in $d\" -ForegroundColor Green
