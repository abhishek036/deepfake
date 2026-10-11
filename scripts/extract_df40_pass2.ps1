Add-Type -AssemblyName System.IO.Compression.FileSystem

$trainDir = "data/raw/DF40/DF40_train"
$zips = Get-ChildItem -Path $trainDir -Filter "*.zip" -File |
        Where-Object { $_.Length -gt 0 } |
        Sort-Object Name

Write-Host "Found $($zips.Count) inner method zips in DF40_train/"
$totalFiles = 0
$errors     = 0

foreach ($zip in $zips) {
    $method = [System.IO.Path]::GetFileNameWithoutExtension($zip.Name)
    $dest   = Join-Path $trainDir $method
    $sizeMB = [math]::Round($zip.Length / 1MB, 1)

    if (Test-Path $dest) {
        $existing = (Get-ChildItem $dest -Recurse -File -ErrorAction SilentlyContinue).Count
        if ($existing -gt 0) {
            Write-Host "SKIP  $method  ($existing files already extracted)"
            continue
        }
    }

    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    Write-Host "EXTRACTING  $($zip.Name)  ($sizeMB MB)  ->  $method/" -ForegroundColor Yellow

    try {
        [System.IO.Compression.ZipFile]::ExtractToDirectory($zip.FullName, $dest)
        $n = (Get-ChildItem $dest -Recurse -File -ErrorAction SilentlyContinue).Count
        Write-Host "  OK  $n files" -ForegroundColor Green
        $totalFiles += $n
    } catch {
        Write-Host "  ERROR: $_" -ForegroundColor Red
        $errors++
    }
}

Write-Host ""
Write-Host "======================================="
Write-Host " Pass-2 Complete"
Write-Host "======================================="
Write-Host " Total files extracted : $totalFiles"
Write-Host " Errors                : $errors"
Write-Host ""

Write-Host "Method folders now in DF40_train/:" -ForegroundColor Cyan
Get-ChildItem -Path $trainDir -Directory | Sort-Object Name | ForEach-Object {
    $n = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue).Count
    Write-Host ("  {0,-25} {1,6} files" -f $_.Name, $n)
}

if ($errors -gt 0) { exit 1 } else { exit 0 }
