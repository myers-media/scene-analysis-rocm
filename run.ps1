param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Resolve-Python {
    $candidates = @(
        @("py", "-3.12"),
        @("py", "-3.11"),
        @("py", "-3.10"),
        @("python")
    )
    foreach ($parts in $candidates) {
        $file = $parts[0]
        $cmd = Get-Command $file -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $args = @()
        if ($parts.Count -gt 1) { $args = $parts[1..($parts.Count - 1)] }
        $code = "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
        & $cmd.Source @($args + @("-c", $code)) | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return @{ Exe = $cmd.Source; Prefix = $args }
        }
    }
    throw "Python 3.10+ is required. PyTorch ROCm wheels target 3.10-3.12."
}

if (-not (Test-Path ".venv")) {
    $py = Resolve-Python
    & $py.Exe @($py.Prefix + @("-m", "venv", ".venv"))
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe scripts\install_torch.py --run
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe -m streamlit run app.py --server.port $Port
