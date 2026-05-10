$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$htmlPath = Join-Path $projectDir "project_poster.html"
$pdfPath = Join-Path $projectDir "project_poster.pdf"
$profilePath = Join-Path $projectDir ".edge-poster-profile"

$edgePaths = @(
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
)

$edge = $edgePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edge) {
    throw "Microsoft Edge was not found. Install Edge or update this script with your browser path."
}

if (-not (Test-Path $htmlPath)) {
    throw "project_poster.html was not found in $projectDir"
}

$htmlUri = (New-Object System.Uri($htmlPath)).AbsoluteUri
$startedAt = Get-Date

if (Test-Path $pdfPath) {
    Remove-Item -LiteralPath $pdfPath -Force
}

& $edge `
    --headless=new `
    --disable-gpu `
    --no-first-run `
    --no-default-browser-check `
    --disable-crash-reporter `
    --disable-crashpad `
    "--user-data-dir=$profilePath" `
    --print-to-pdf-no-header `
    "--print-to-pdf=$pdfPath" `
    $htmlUri

Start-Sleep -Milliseconds 500

if (-not (Test-Path $pdfPath)) {
    $fallbackPdf = Get-ChildItem -LiteralPath $projectDir -Filter "*.pdf" -File |
        Where-Object { $_.LastWriteTime -ge $startedAt.AddSeconds(-2) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($fallbackPdf) {
        Move-Item -LiteralPath $fallbackPdf.FullName -Destination $pdfPath -Force
    } else {
        throw "PDF export failed. Expected output was not created: $pdfPath"
    }
}

if (Test-Path $profilePath) {
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            Remove-Item -LiteralPath $profilePath -Recurse -Force
            break
        } catch {
            if ($attempt -eq 5) {
                throw
            }
            Start-Sleep -Milliseconds 500
        }
    }
}

Write-Host "Saved PDF:" $pdfPath
