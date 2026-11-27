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
        $supported = & $PythonPath -c "import sys; print(int((3, 10) <= sys.version_info[:2] < (3, 14)))" 2>$null
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
        foreach ($version in @("3.12", "3.13", "3.11", "3.10")) {
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
    throw "Python 3.10-3.13 was not found. Install 64-bit Python 3.12 with Add Python to PATH enabled."
}

function Invoke-Python([string]$PythonPath, [string[]]$Arguments) {
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit code $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

function Get-FileSha256([string]$Path) {
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $bytes = $sha256.ComputeHash($stream)
        return ([System.BitConverter]::ToString($bytes)).Replace("-", "")
    }
    finally {
        $stream.Dispose()
        $sha256.Dispose()
    }
}

function Get-DependencyState {
    $requirementsHash = Get-FileSha256 $requirements
    $projectHash = Get-FileSha256 $projectFile
    return "$requirementsHash`n$projectHash`n$script:TorchBackend"
}

function Get-TorchBackend {
    $override = $env:DATASET_MANAGER_TORCH_BACKEND
    if ($override) {
        $normalized = $override.Trim().ToLowerInvariant()
        if ($normalized -notin @("cpu", "cu128", "cu132")) {
            throw "DATASET_MANAGER_TORCH_BACKEND must be cpu, cu128, or cu132."
        }
        return $normalized
    }

    $nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return "cpu"
    }
    try {
        $smiOutput = (& $nvidiaSmi.Source 2>$null) -join "`n"
        if ($LASTEXITCODE -ne 0) {
            return "cpu"
        }
        $match = [regex]::Match(
            $smiOutput,
            "CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)",
            [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
        )
        if (-not $match.Success) {
            Write-Warning "NVIDIA GPU detected, but the supported CUDA version could not be read. Using CPU PyTorch."
            return "cpu"
        }
        $cudaVersion = [version]::new(
            [int]$match.Groups[1].Value,
            [int]$match.Groups[2].Value
        )
        if ($cudaVersion -ge [version]"13.2") {
            return "cu132"
        }
        if ($cudaVersion -ge [version]"12.8") {
            return "cu128"
        }
        Write-Warning "NVIDIA driver supports CUDA $cudaVersion, below the packaged CUDA 12.8 runtime. Using CPU PyTorch."
        return "cpu"
    }
    catch {
        Write-Warning "NVIDIA detection failed: $($_.Exception.Message). Using CPU PyTorch."
        return "cpu"
    }
}

function Install-TorchBackend([string]$Backend) {
    Write-Host "[2/4] Synchronizing PyTorch backend: $Backend..."
    switch ($Backend) {
        "cu132" {
            Invoke-Python $venvPython @(
                "-m", "pip", "install", "--disable-pip-version-check", "--upgrade",
                "torch==2.15.0+cu132", "torchvision==0.29.2+cu132",
                "--index-url", "https://download.pytorch.org/whl/cu132"
            )
        }
        "cu128" {
            Invoke-Python $venvPython @(
                "-m", "pip", "install", "--disable-pip-version-check", "--upgrade",
                "torch==2.11.0+cu128", "torchvision==0.26.0+cu128",
                "--index-url", "https://download.pytorch.org/whl/cu128"
            )
        }
        default {
            Invoke-Python $venvPython @(
                "-m", "pip", "install", "--disable-pip-version-check", "--upgrade",
                "torch==2.15.0", "torchvision==0.29.2",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            )
        }
    }
}

function Test-TorchBackend([string]$Backend) {
    $details = & $venvPython -c (
        "import torch; " +
        "print(torch.__version__); " +
        "print(torch.version.cuda or 'none'); " +
        "print(int(torch.cuda.is_available())); " +
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch backend verification failed."
    }
    $lines = @($details)
    if ($lines.Count -lt 4) {
        throw "PyTorch backend verification returned incomplete information."
    }
    Write-Host "[4/4] PyTorch $($lines[0]); CUDA runtime $($lines[1]); device $($lines[3])"
    if ($Backend -ne "cpu" -and $lines[2].Trim() -ne "1") {
        throw "CUDA PyTorch was installed, but torch.cuda.is_available() is False. Update the NVIDIA driver or set DATASET_MANAGER_TORCH_BACKEND=cpu."
    }
}

Write-Host "[Dataset Manager] Project: $projectRoot"
$script:TorchBackend = Get-TorchBackend
Write-Host "[Dataset Manager] Compute backend: $script:TorchBackend"

if (-not (Test-Path -LiteralPath $venvPython)) {
    $basePython = Find-BasePython
    $version = & $basePython --version
    Write-Host "[1/4] Creating the local environment with $version..."
    Invoke-Python $basePython @("-m", "venv", $venvRoot)
}
elseif (-not (Test-SupportedPython $venvPython)) {
    throw "The Python version in .venv is unsupported. Delete .venv and run setup_environment.bat again."
}
else {
    Write-Host "[1/4] Reusing local environment: $venvRoot"
}

$expectedState = Get-DependencyState
$currentState = if (Test-Path -LiteralPath $stateFile) {
    (Get-Content -LiteralPath $stateFile -Raw).TrimEnd()
} else {
    ""
}

if ($ForceInstall -or $currentState -ne $expectedState.TrimEnd()) {
    Install-TorchBackend $script:TorchBackend
    Write-Host "[3/4] Installing or synchronizing application dependencies..."
    Invoke-Python $venvPython @(
        "-m", "pip", "install", "--disable-pip-version-check",
        "-r", $requirements
    )
    Invoke-Python $venvPython @(
        "-m", "pip", "install", "--disable-pip-version-check",
        "--no-deps", "--editable", $projectRoot
    )
    Test-TorchBackend $script:TorchBackend
    [System.IO.File]::WriteAllText($stateFile, $expectedState, [System.Text.UTF8Encoding]::new($false))
    Write-Host "[Dataset Manager] Dependencies are ready."
}
else {
    Write-Host "[2/4] Dependencies are unchanged; installation skipped."
    Test-TorchBackend $script:TorchBackend
}

if ($SetupOnly) {
    Write-Host "[Dataset Manager] Environment setup completed."
    exit 0
}

Write-Host "[Dataset Manager] Starting Dataset Manager..."
Push-Location $projectRoot
try {
    Invoke-Python $venvPython @("-m", "dataset_manager")
}
finally {
    Pop-Location
}
