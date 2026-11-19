param(
    [switch]$SetupOnly,
    [switch]$ForceInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvRoot = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$requirements = Join-Path $projectRoot "requirements-runtime.txt"
$projectFile = Join-Path $projectRoot "pyproject.toml"
$stateFile = Join-Path $venvRoot ".dataset-manager-dependencies"

function Test-SupportedPython([string]$PythonPath) {
    try {
        $supported = & $PythonPath -c "import sys; print(int((3, 10) <= sys.version_info[:2] < (3, 15)))" 2>$null
        return $LASTEXITCODE -eq 0 -and $supported.Trim() -eq "1"
    }
    catch {
        return $false
    }
}

function Find-BasePython {
    $candidates = [System.Collections.Generic.List[string]]::new()
    $pyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        foreach ($version in @("3.12", "3.13", "3.14", "3.11", "3.10")) {
            try {
                $path = & $pyLauncher.Source "-$version" -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $path) {
                    $candidates.Add($path.Trim())
                }
            }
            catch {
                # Try the next installed Python version.
            }
        }
    }

    $pythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        try {
            $path = & $pythonCommand.Source -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $path) {
                $candidates.Add($path.Trim())
            }
        }
        catch {
            # The Windows Store alias may exist without an installed interpreter.
        }
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if ((Test-Path -LiteralPath $candidate) -and (Test-SupportedPython $candidate)) {
            return $candidate
        }
    }
    throw "Python 3.10-3.14 was not found. Install 64-bit Python 3.12 with Add Python to PATH enabled."
}

function Invoke-Python([string]$PythonPath, [string[]]$Arguments) {
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit code $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

function Get-DependencyState {
    $requirementsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $requirements).Hash
    $projectHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $projectFile).Hash
    return "$requirementsHash`n$projectHash"
}

Write-Host "[Dataset Manager] Project: $projectRoot"

if (-not (Test-Path -LiteralPath $venvPython)) {
    $basePython = Find-BasePython
    $version = & $basePython --version
    Write-Host "[1/3] Creating the local environment with $version..."
    Invoke-Python $basePython @("-m", "venv", $venvRoot)
}
elseif (-not (Test-SupportedPython $venvPython)) {
    throw "The Python version in .venv is unsupported. Delete .venv and run setup_environment.bat again."
}
else {
    Write-Host "[1/3] Reusing local environment: $venvRoot"
}

$expectedState = Get-DependencyState
$currentState = if (Test-Path -LiteralPath $stateFile) {
    (Get-Content -LiteralPath $stateFile -Raw).TrimEnd()
} else {
    ""
}

if ($ForceInstall -or $currentState -ne $expectedState.TrimEnd()) {
    Write-Host "[2/3] Installing or synchronizing dependencies (the first run needs network access)..."
    Invoke-Python $venvPython @(
        "-m", "pip", "install", "--disable-pip-version-check",
        "-r", $requirements
    )
    Invoke-Python $venvPython @(
        "-m", "pip", "install", "--disable-pip-version-check",
        "--no-deps", "--editable", $projectRoot
    )
    [System.IO.File]::WriteAllText($stateFile, $expectedState, [System.Text.UTF8Encoding]::new($false))
    Write-Host "[2/3] Dependencies are ready."
}
else {
    Write-Host "[2/3] Dependencies are unchanged; installation skipped."
}

if ($SetupOnly) {
    Write-Host "[3/3] Environment setup completed."
    exit 0
}

Write-Host "[3/3] Starting Dataset Manager..."
Push-Location $projectRoot
try {
    Invoke-Python $venvPython @("-m", "dataset_manager")
}
finally {
    Pop-Location
}
