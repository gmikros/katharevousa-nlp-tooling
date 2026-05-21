# Sync local changes with GitHub for katharevousa-nlp-tooling.
# Usage:
#   .\scripts\sync_github.ps1 status
#   .\scripts\sync_github.ps1 pull
#   .\scripts\sync_github.ps1 push -Message "Update paper results"
param(
    [Parameter(Position = 0)]
    [ValidateSet("status", "pull", "push")]
    [string]$Action = "status",

    [string]$Message = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

function Assert-GitRepo {
    if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
        throw "Not a git repository: $RepoRoot"
    }
}

function Show-Remote {
    git remote -v
    Write-Host ""
    git status -sb
}

Assert-GitRepo

switch ($Action) {
    "status" {
        Show-Remote
    }
    "pull" {
        Write-Host "Fetching and merging from origin/main..."
        git fetch origin
        git pull --rebase origin main
        Show-Remote
    }
    "push" {
        if (-not $Message) {
            throw "Provide -Message when pushing, e.g. .\scripts\sync_github.ps1 push -Message `"Update paper`""
        }
        git add -A
        git status -sb
        git commit -m $Message
        git push origin main
        Show-Remote
    }
}
