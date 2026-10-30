ese @echo off
echo ========================================================
echo Pushing Retail Demand Forecasting Project to GitHub...
echo ========================================================

if not exist ".git" (
    git init
)

git config user.name "Yash Sidana"
git config user.email "yashsidana@users.noreply.github.com"

git remote remove origin 2>nul
git remote add origin https://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@github.com/yashsidana/Retail_demand.git

git add .
git commit -m "feat: end-to-end retail demand forecasting & inventory optimization system"
git branch -M main
git push -u origin main --force

echo.
echo ========================================================
echo Push complete! Repository: https://github.com/yashsidana/Retail_demand
echo ========================================================
pause

