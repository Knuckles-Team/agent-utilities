<#
.SYNOPSIS
  One-link self-deploy bootstrap for agent-utilities on Windows.

.DESCRIPTION
  The Windows counterpart of scripts/install.sh. Checks Python (>=3.11,<3.15),
  ensures uv, installs agent-utilities (+ universal-skills), runs the host
  dependency preflight for the chosen profile, installs the skill toolkit into
  every agent tool present, wires the graph-os MCP server into each, then hands
  off to the guided deployment skill.

  It does NOT fabricate the *-mcp fleet config or secrets — that is the
  agent-utilities-deployment skill's job (it reads genesis.yaml).

.EXAMPLE
  # From a clone:
  .\scripts\install.ps1 -DeployProfile single-node-prod -Component agent-webui

#>
[CmdletBinding()]
param(
  [ValidateSet('tiny', 'single-node-prod', 'enterprise')]
  [string]$DeployProfile = $(if ($env:AU_PROFILE) { $env:AU_PROFILE } else { 'tiny' }),
  [string[]]$Component = @(),
  [ValidateSet('', 'all', 'none')]
  [string]$Extras = '',
  [switch]$Editable,
  [switch]$NoSkills,
  [switch]$NoMcp,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "[install] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[install] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[install] $msg" -ForegroundColor Red }

function Invoke-Step {
  param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [string[]]$Arguments = @()
  )
  if ($DryRun) {
    $rendered = @($Executable) + ($Arguments | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' })
    Write-Host "  > $($rendered -join ' ')"
    return
  }
  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) { throw "command failed" }
}

# Components from env (comma-separated) merged with -Component.
if ($env:AU_COMPONENTS) { $Component += $env:AU_COMPONENTS.Split(',') }
$Component = $Component | Where-Object { $_ }

# Auto extras: full integration unless the zero-infra tiny profile.
if (-not $Extras) { $Extras = if ($DeployProfile -eq 'tiny') { 'none' } else { 'all' } }

Write-Step "profile=$DeployProfile  components=[$($Component -join ',')]  extras=$Extras  dry_run=$DryRun"

# 1. Python check (the package enforces >=3.11,<3.15). Windows PowerShell 5.1-safe.
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $py) { Write-Err 'Python not found — install Python 3.11-3.14 (https://python.org).'; exit 1 }
$pyExe = $py.Source
$verOk = & $pyExe -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] < (3,15) else 1)'; $rc = $LASTEXITCODE
$pyVer = & $pyExe -c 'import sys; print("%d.%d" % sys.version_info[:2])'
Write-Step "Python $pyVer"
if ($rc -ne 0) { Write-Err 'agent-utilities needs Python >=3.11,<3.15.'; exit 1 }

# 2. Require a separately verified uv installation. Never execute a downloaded
# script from a privileged bootstrap.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Err 'uv not found — install a verified uv release, then rerun this script.'
  exit 1
}

# Published bootstraps are exact at the project boundary. Release automation
# updates these defaults; controlled mirrors may inject another exact version.
$auVersion = if ($env:AGENT_UTILITIES_VERSION) { $env:AGENT_UTILITIES_VERSION } else { '1.27.1' }
$skillsVersion = if ($env:UNIVERSAL_SKILLS_VERSION) { $env:UNIVERSAL_SKILLS_VERSION } else { '1.2.1' }
$versionPattern = '^[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9.+-]*$'
if ($auVersion -notmatch $versionPattern -or $skillsVersion -notmatch $versionPattern) {
  Write-Err 'invalid exact package version'
  exit 2
}
$pkg = if ($Extras -eq 'all') { "agent-utilities[all]==$auVersion" } else { "agent-utilities==$auVersion" }

# Detect running inside a clone (editable install path).
$repoRoot = Split-Path -Parent $PSScriptRoot
$inRepo = $false
if ($repoRoot -and (Test-Path "$repoRoot\pyproject.toml")) {
  if (Select-String -Path "$repoRoot\pyproject.toml" -Pattern 'name = "agent-utilities"' -Quiet) { $inRepo = $true }
}

Write-Step "Installing $pkg + universal-skills..."
if ($Editable -and $inRepo) {
  $editableSpec = if ($Extras -eq 'all') { "$repoRoot[all]" } else { $repoRoot }
  Invoke-Step 'uv' @('pip', 'install', '--system', '-e', $editableSpec)
} else {
  Invoke-Step 'uv' @('tool', 'install', $pkg)
  Invoke-Step 'uv' @('tool', 'install', "universal-skills==$skillsVersion")
}

# 3. Host dependency preflight.
$pfArgs = @('--preflight', '--profile', $DeployProfile)
foreach ($c in $Component) { $pfArgs += @('--component', $c) }
Write-Step 'Running host preflight...'
if (Get-Command agent-utilities-doctor -ErrorAction SilentlyContinue) {
  Invoke-Step 'agent-utilities-doctor' $pfArgs
} else {
  Invoke-Step $pyExe (@('-m', 'agent_utilities.deployment.doctor') + $pfArgs)
}

# 4. Install the skill toolkit into every agent tool present, wire graph-os MCP.
if (-not $NoSkills) {
  Write-Step 'Installing skills into every detected agent tool (--all-detected)...'
  if (Get-Command install-skills -ErrorAction SilentlyContinue) {
    Invoke-Step 'install-skills' @('--all-detected', '--symlink')
  } else {
    Write-Warn 'install-skills not on PATH — skipping (pip install universal-skills).'
  }
}

if (-not $NoMcp) {
  if ((Get-Command codex -ErrorAction SilentlyContinue) -and (Get-Command setup-config -ErrorAction SilentlyContinue)) {
    Write-Step 'Registering the portable graph-os stdio launcher with Codex...'
    Invoke-Step 'setup-config' @('codex')
  }
  Write-Step 'Wiring graph-os into detected JSON-config MCP clients...'
  $mcpSrc = Join-Path $env:TEMP "graphos-mcp-$([System.IO.Path]::GetRandomFileName()).json"
  @'
{
  "mcpServers": {
    "graph-os": { "command": "graph-os", "args": ["--transport", "stdio"] }
  }
}
'@ | Set-Content -Path $mcpSrc -Encoding utf8
  $mcpInstall = & $pyExe -c 'import importlib.util as u,os;s=u.find_spec("universal_skills");print(os.path.join(os.path.dirname(s.origin),"agent-tools","mcp-installer","scripts","install.py") if s else "")'
  if ($mcpInstall -and (Test-Path $mcpInstall)) {
    Invoke-Step $pyExe @($mcpInstall, '--config', $mcpSrc, '--all-detected')
  } else {
    Write-Warn 'mcp-installer not found — skipping graph-os MCP wiring.'
  }
  if (-not $DryRun) { Remove-Item $mcpSrc -ErrorAction SilentlyContinue }
}

# 5. Hand off to the deployment skill.
$skill = 'agent-utilities-deployment'
Write-Host ''
Write-Step 'Done. Your agent now has the KG tools + the genesis skills.'
Write-Host @"

Next — open your agent (Claude Code / Cursor / ...) and say:

    deploy agent-utilities (profile=$DeployProfile)

It will run the '$skill' skill, read genesis.yaml, generate the full config
(setup-config), wire the *-mcp fleet + any UI components, and verify with
'agent-utilities-doctor'.

Manual path:
    setup-config generate --profile $DeployProfile
    setup-config codex             # native Codex config.toml registration
    graph-os                       # stdio MCP for your IDE
    graph-os-daemon                # REST gateway (:8100)
    agent-utilities-doctor         # verify

Docs: https://knuckles-team.github.io/agent-utilities/
"@
