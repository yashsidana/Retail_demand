# PowerShell Git Setup and Push Script
$ErrorActionPreference = "Stop"

Write-Host "Initializing Git Repository..." -ForegroundColor Cyan
if (-not (Test-Path ".git")) {
    git init
}

git config user.name "Yash Sidana"
git config user.email "yashsidana@users.noreply.github.com"

Write-Host "Configuring remote origin with provided access token..." -ForegroundColor Cyan
$repoUrl = "https://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@github.com/yashsidana/Retail_demand.git"

# Check if origin already exists
$remotes = git remote
if ($remotes -contains "origin") {
    git remote set-url origin $repoUrl
} else {
    git remote add origin $repoUrl
}

Write-Host "Staging project files..." -ForegroundColor Cyan
git add .

Write-Host "Creating commit..." -ForegroundColor Cyan
git commit -m "feat: end-to-end retail demand forecasting & inventory optimization system"

Write-Host "Setting main branch..." -ForegroundColor Cyan
git branch -M main

Write-Host "Pushing to GitHub (yashsidana/Retail_demand)..." -ForegroundColor Green
git push -u origin main --force

Write-Host "`nSUCCESS! Codebase pushed to https://github.com/yashsidana/Retail_demand" -ForegroundColor Green

