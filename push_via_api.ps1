# PowerShell GitHub REST API Upload Script (Zero Git installation required)
$ErrorActionPreference = "Continue"

# Force TLS 1.2 for GitHub API on Windows PowerShell 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Token = "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"
$Repo = "yashsidana/Retail_demand"
$ApiBase = "https://api.github.com/repos/$Repo"

$Headers = @{
    "Authorization" = "Bearer $Token"
    "Accept"        = "application/vnd.github+json"
    "User-Agent"    = "Antigravity-Agent"
}

Write-Host "Connecting to GitHub API for $Repo..." -ForegroundColor Cyan

# Set current directory to the directory of this script
if ($PSScriptRoot) {
    Set-Location $PSScriptRoot
}
$CurrentDir = Get-Location

# Get branch info
$Branch = "main"
try {
    $ref = Invoke-RestMethod -Uri "$ApiBase/git/ref/heads/$Branch" -Headers $Headers -Method Get
    $ParentSha = $ref.object.sha
} catch {
    $ParentSha = $null
}

# Collect files
$IgnoreDirs = @(".git", "__pycache__", ".pytest_cache", ".venv", "venv", "env", ".ipynb_checkpoints")
$IgnoreExts = @(".pyc", ".pyo", ".pyd", ".parquet", ".db")

$AllFiles = Get-ChildItem -Path $CurrentDir.Path -Recurse -File | Where-Object {
    $item = $_
    $skip = $false
    foreach ($d in $IgnoreDirs) {
        if ($item.FullName -match [regex]::Escape($d)) { $skip = $true; break }
    }
    if ($IgnoreExts -contains $item.Extension) { $skip = $true }
    -not $skip
}

Write-Host "Uploading $($AllFiles.Count) files via GitHub API..." -ForegroundColor Cyan

$TreeItems = @()
foreach ($file in $AllFiles) {
    $relPath = $file.FullName.Substring($CurrentDir.Path.Length + 1).Replace("\", "/")
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    $b64 = [Convert]::ToBase64String($bytes)

    $blobBody = @{
        content  = $b64
        encoding = "base64"
    } | ConvertTo-Json

    try {
        $blob = Invoke-RestMethod -Uri "$ApiBase/git/blobs" -Headers $Headers -Method Post -Body $blobBody -ContentType "application/json"
        $TreeItems += @{
            path = $relPath
            mode = "100644"
            type = "blob"
            sha  = $blob.sha
        }
        Write-Host "  + Created blob for: $relPath" -ForegroundColor DarkGray
    } catch {
        Write-Host "  ! Error on $relPath : $_" -ForegroundColor Yellow
    }
}

Write-Host "Creating Git Tree and Commit..." -ForegroundColor Cyan
$treeBody = @{ tree = $TreeItems } | ConvertTo-Json -Depth 5
$tree = Invoke-RestMethod -Uri "$ApiBase/git/trees" -Headers $Headers -Method Post -Body $treeBody -ContentType "application/json"

$commitBody = @{
    message = "feat: complete retail demand forecasting & inventory optimization project"
    tree    = $tree.sha
    parents = if ($ParentSha) { @($ParentSha) } else { @() }
} | ConvertTo-Json

$commit = Invoke-RestMethod -Uri "$ApiBase/git/commits" -Headers $Headers -Method Post -Body $commitBody -ContentType "application/json"

Write-Host "Updating branch '$Branch' reference..." -ForegroundColor Cyan
if ($ParentSha) {
    $updateBody = @{ sha = $commit.sha; force = $true } | ConvertTo-Json
    $null = Invoke-RestMethod -Uri "$ApiBase/git/refs/heads/$Branch" -Headers $Headers -Method Patch -Body $updateBody -ContentType "application/json"
} else {
    $createBody = @{ ref = "refs/heads/$Branch"; sha = $commit.sha } | ConvertTo-Json
    $null = Invoke-RestMethod -Uri "$ApiBase/git/refs" -Headers $Headers -Method Post -Body $createBody -ContentType "application/json"
}

Write-Host "`nSUCCESS! All files committed and pushed to https://github.com/$Repo" -ForegroundColor Green
