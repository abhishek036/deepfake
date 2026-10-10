# ============================================================
# scripts/extract_df40.ps1
# Extract all DF40 zip archives into data/raw/DF40/
#   Pass 1: outer zips at root level  (DF40_train-*.zip, inswap-017.zip, ...)
#   Pass 2: inner zips inside DF40_train/  (DiT.zip, fomm.zip, ...)
# ============================================================
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File scripts\extract_df40.ps1
# ============================================================

param(
    [string]$DF40Root  = "C:\deepfake\data\raw\DF40",
    [switch]$DryRun    = $false,
    [switch]$Pass2Only = $false   # skip pass-1, only extract DF40_train inner zips
)

$ErrorActionPreference = "Continue"   # don't abort on single-zip errors
Add-Type -AssemblyName System.IO.Compression.FileSystem

# ── Helper: extract one zip to a destination folder ──────────────────────────
function Expand-ZipTo {
    param(
        [string]$ZipPath,
        [string]$DestDir,
        [ref]$Extracted,
        [ref]$Skipped
    )

    $archive = $null
    try {
        $archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
        foreach ($entry in $archive.Entries) {
            if ($entry.FullName.EndsWith('/') -or $entry.FullName.EndsWith('\')) { continue }
            $dest    = Join-Path $DestDir $entry.FullName
            $destDir = Split-Path $dest -Parent
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)) {
                $Skipped.Value++
                continue
            }
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
            $Extracted.Value++
        }
        $archive.Dispose()
        return $true
    } catch {
        if ($archive) { try { $archive.Dispose() } catch {} }
        throw $_
    }
}

# ── Resolve root ──────────────────────────────────────────────────────────────
$resolved = Resolve-Path $DF40Root -ErrorAction SilentlyContinue
if (-not $resolved) { Write-Error "DF40 root not found: $DF40Root"; exit 1 }
$DF40Root = $resolved.Path

Write-Host "`n=====================================================" -ForegroundColor Cyan
Write-Host "  FORGE-Lite | DF40 Two-Pass Extraction" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Root : $DF40Root"
Write-Host "  Mode : $(if ($DryRun) { 'DRY RUN' } else { 'LIVE' })"
Write-Host "=====================================================`n"

$grandExtracted = 0
$grandSkipped   = 0
$grandErrors    = 0
$startTime      = Get-Date

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — outer zips at DF40 root level (skip if -Pass2Only)
# ══════════════════════════════════════════════════════════════════════════════
if (-not $Pass2Only) {
    $outerZips = Get-ChildItem -Path $DF40Root -Filter "*.zip" -File |
                 Where-Object { $_.Name -notlike "*.part" -and $_.Length -gt 0 } |
                 Sort-Object Name

    Write-Host "PASS 1 — $($outerZips.Count) outer zip(s) at root level`n" -ForegroundColor Cyan

    foreach ($zip in $outerZips) {
        $sizeMB = [math]::Round($zip.Length / 1MB, 1)
        Write-Host "  -- $($zip.Name)  ($sizeMB MB)" -ForegroundColor Yellow

        if ($DryRun) { Write-Host "     [DRY RUN] skipped`n"; continue }

        $ex = 0; $sk = 0
        try {
            Expand-ZipTo -ZipPath $zip.FullName -DestDir $DF40Root -Extracted ([ref]$ex) -Skipped ([ref]$sk)
            $grandExtracted += $ex; $grandSkipped += $sk
            Write-Host "     Extracted: $ex | Skipped: $sk" -ForegroundColor Green
        } catch {
            $grandErrors++
            Write-Host "     ERROR: $_" -ForegroundColor Red
        }
    }
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — inner zips inside DF40_train/ (the real method datasets)
# ══════════════════════════════════════════════════════════════════════════════
$trainDir = Join-Path $DF40Root "DF40_train"
if (Test-Path $trainDir) {
    $innerZips = Get-ChildItem -Path $trainDir -Filter "*.zip" -File |
                 Where-Object { $_.Length -gt 0 } |
                 Sort-Object Name

    if ($innerZips.Count -gt 0) {
        Write-Host "PASS 2 — $($innerZips.Count) inner method zip(s) inside DF40_train/`n" -ForegroundColor Cyan

        foreach ($zip in $innerZips) {
            # Each inner zip should extract into DF40_train/<method>/
            $methodName = [System.IO.Path]::GetFileNameWithoutExtension($zip.Name)
            $methodDest = Join-Path $trainDir $methodName
            $sizeMB     = [math]::Round($zip.Length / 1MB, 1)

            Write-Host "  -- $($zip.Name)  ($sizeMB MB)  ->  DF40_train\$methodName\" -ForegroundColor Yellow

            if ($DryRun) { Write-Host "     [DRY RUN] skipped`n"; continue }

            # Skip if already extracted (folder exists and has files)
            if ((Test-Path $methodDest) -and
                ((Get-ChildItem $methodDest -Recurse -File -ErrorAction SilentlyContinue).Count -gt 0)) {
                Write-Host "     Already extracted — skipped" -ForegroundColor DarkGray
                $grandSkipped++
                continue
            }

            New-Item -ItemType Directory -Path $methodDest -Force | Out-Null
            $ex = 0; $sk = 0
            try {
                Expand-ZipTo -ZipPath $zip.FullName -DestDir $methodDest -Extracted ([ref]$ex) -Skipped ([ref]$sk)
                $grandExtracted += $ex; $grandSkipped += $sk
                Write-Host "     Extracted: $ex | Skipped: $sk" -ForegroundColor Green
            } catch {
                $grandErrors++
                Write-Host "     ERROR: $_" -ForegroundColor Red
            }
        }
        Write-Host ""
    } else {
        Write-Host "PASS 2 — No inner zips found in DF40_train/ (already extracted or empty)`n" -ForegroundColor Gray
    }
} else {
    Write-Host "PASS 2 — DF40_train/ not found; skipping inner extraction`n" -ForegroundColor DarkYellow
}

# ── Summary ───────────────────────────────────────────────────────────────────
$elapsed = (Get-Date) - $startTime
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Extraction Complete" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Files extracted : $grandExtracted"
Write-Host "  Files skipped   : $grandSkipped"
Write-Host "  Errors          : $grandErrors"
Write-Host "  Elapsed         : $([math]::Round($elapsed.TotalMinutes, 1)) min"
Write-Host "=====================================================" -ForegroundColor Cyan

# ── Final folder inventory ────────────────────────────────────────────────────
Write-Host "`nMethod folders now in DF40_train/ :" -ForegroundColor Cyan
if (Test-Path $trainDir) {
    Get-ChildItem -Path $trainDir -Directory | Sort-Object Name | ForEach-Object {
        $n = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue).Count
        Write-Host ("  {0,-20} {1,6} files" -f "$($_.Name)/", $n)
    }
}

Write-Host "`nOther top-level method folders :" -ForegroundColor Cyan
Get-ChildItem -Path $DF40Root -Directory |
    Where-Object { $_.Name -ne "DF40_train" } |
    Sort-Object Name | ForEach-Object {
        $n = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue).Count
        Write-Host ("  {0,-20} {1,6} files" -f "$($_.Name)/", $n)
    }

Write-Host ""

# ── pixart warning ────────────────────────────────────────────────────────────
$pixartZip = Join-Path $DF40Root "pixart-002.zip"
if ((Test-Path $pixartZip) -and (Get-Item $pixartZip).Length -eq 0) {
    Write-Host "WARNING: pixart-002.zip is 0 bytes (corrupt/incomplete download)." -ForegroundColor Red
    Write-Host "         Re-download pixart-002.zip from the DF40 Google Drive link." -ForegroundColor Red
    Write-Host "         All other methods are available without it.`n" -ForegroundColor Red
}

if ($grandErrors -gt 0) { exit 1 } else { exit 0 }
