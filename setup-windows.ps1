#Requires -Version 5.0

################################################################################
# Agent Platform Setup Script for Windows (PowerShell)
# Version: 1.0
# Supports: Windows 10/11, PowerShell 5.0+
################################################################################

param(
    [switch]$SkipDocker = $false,
    [switch]$SkipPrerequisiteCheck = $false
)

# Script configuration
$ErrorActionPreference = "Stop"
$WarningPreference = "Continue"
$VerbosePreference = "SilentlyContinue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectName = "Agent Platform"
$PythonMinVersion = "3.10"
$NodeMinVersion = 18

################################################################################
# UTILITY FUNCTIONS
################################################################################

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
    Write-Host "║ $Message" -ForegroundColor Blue
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Cyan
}

function Test-CommandExists {
    param([string]$Command)
    try {
        if (Get-Command $Command -ErrorAction Stop) {
            return $true
        }
    }
    catch {
        return $false
    }
}

function Get-CommandVersion {
    param([string]$Command)
    try {
        if (Test-CommandExists $Command) {
            & $Command --version 2>$null | Select-Object -First 1
        }
        else {
            return "not installed"
        }
    }
    catch {
        return "unknown"
    }
}

################################################################################
# PREREQUISITE CHECKS
################################################################################

function Test-Prerequisites {
    Write-Header "Checking Prerequisites"
    
    $missingDeps = $false
    
    # Check Python
    if (Test-CommandExists python) {
        $pyVersion = & python --version 2>&1 | Select-String -Pattern '\d+\.\d+' -AllMatches | ForEach-Object { $_.Matches[0].Value }
        Write-Success "Python $pyVersion installed"
    }
    elseif (Test-CommandExists python3) {
        $pyVersion = & python3 --version 2>&1 | Select-String -Pattern '\d+\.\d+' -AllMatches | ForEach-Object { $_.Matches[0].Value }
        Write-Success "Python $pyVersion installed"
    }
    else {
        Write-Error-Custom "Python 3 not found"
        $missingDeps = $true
    }
    
    # Check Node.js
    if (Test-CommandExists node) {
        $nodeVersion = & node --version
        Write-Success "Node.js $nodeVersion installed"
    }
    else {
        Write-Error-Custom "Node.js not found"
        $missingDeps = $true
    }
    
    # Check Git
    if (Test-CommandExists git) {
        $gitVersion = & git --version | ForEach-Object { $_ -split '\s+' | Select-Object -Last 1 }
        Write-Success "Git installed"
    }
    else {
        Write-Warning-Custom "Git not found (recommended)"
    }
    
    if ($missingDeps) {
        Write-Error-Custom "Missing required dependencies"
        Write-Host ""
        Write-Host "Installation instructions for Windows:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Option 1: Using Chocolatey (Recommended)"
        Write-Host "  choco install python nodejs git"
        Write-Host ""
        Write-Host "Option 2: Using Winget"
        Write-Host "  winget install Python.Python.3.11"
        Write-Host "  winget install OpenJS.NodeJS"
        Write-Host "  winget install Git.Git"
        Write-Host ""
        Write-Host "Option 3: Manual Installation"
        Write-Host "  Python: https://www.python.org/downloads/"
        Write-Host "  Node.js: https://nodejs.org/en/download/"
        Write-Host "  Git: https://git-scm.com/download/win"
        Write-Host ""
        
        return $false
    }
    
    return $true
}

################################################################################
# PYTHON ENVIRONMENT SETUP
################################################################################

function Initialize-PythonEnvironment {
    Write-Header "Setting Up Python Environment"
    
    $venvPath = Join-Path $ScriptDir "venv"
    
    if (Test-Path $venvPath) {
        Write-Warning-Custom "Virtual environment already exists at $venvPath"
        $response = Read-Host "Recreate it? (y/n)"
        
        if ($response -eq "y") {
            Write-Info "Removing old virtual environment..."
            Remove-Item $venvPath -Recurse -Force
        }
        else {
            Write-Info "Using existing virtual environment"
            return $true
        }
    }
    
    Write-Info "Creating virtual environment..."
    & python -m venv $venvPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to create virtual environment"
        return $false
    }
    
    Write-Success "Virtual environment created"
    
    # Activate venv
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    & $activateScript
    
    Write-Info "Upgrading pip, setuptools, wheel..."
    & python -m pip install --upgrade pip setuptools wheel | Out-Null
    
    Write-Success "Python environment ready"
    return $true
}

function Install-PythonDependencies {
    Write-Header "Installing Python Dependencies"
    
    $requirementsFile = Join-Path $ScriptDir "requirements.txt"
    
    if (-not (Test-Path $requirementsFile)) {
        Write-Error-Custom "requirements.txt not found at $requirementsFile"
        return $false
    }
    
    Write-Info "Installing packages from requirements.txt..."
    Write-Info "This may take several minutes..."
    
    & pip install -r $requirementsFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to install Python dependencies"
        return $false
    }
    
    Write-Success "Python dependencies installed"
    return $true
}

################################################################################
# NODE DEPENDENCIES SETUP
################################################################################

function Install-NodeDependencies {
    Write-Header "Installing Node.js Dependencies"
    
    Write-Info "Installing root dependencies..."
    & npm install
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to install root dependencies"
        return $false
    }
    
    Write-Info "Installing frontend dependencies..."
    $frontendPath = Join-Path $ScriptDir "frontend"
    Push-Location $frontendPath
    & npm install
    Pop-Location
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to install frontend dependencies"
        return $false
    }
    
    Write-Success "Node.js dependencies installed"
    return $true
}

################################################################################
# CONFIGURATION SETUP
################################################################################

function Initialize-EnvironmentFile {
    Write-Header "Setting Up Environment Configuration"
    
    $envFile = Join-Path $ScriptDir ".env"
    $envExample = Join-Path $ScriptDir ".env.example"
    
    if (Test-Path $envFile) {
        Write-Warning-Custom ".env file already exists"
        $response = Read-Host "Overwrite with template? (y/n)"
        
        if ($response -ne "y") {
            Write-Info "Keeping existing .env file"
            return $true
        }
    }
    
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile -Force
        Write-Success ".env file created from .env.example"
    }
    else {
        # Create default .env file
        $envContent = @"
# Agent Platform Environment Variables

# LLM Provider Configuration (choose at least one)
# GROQ_API_KEY=gsk_...
OLLAMA_BASE_URL=http://localhost:11434
# OPENAI_API_KEY=sk_...
# GEMINI_API_KEY=AIza...
# ANTHROPIC_API_KEY=sk-ant-...

# Search Configuration
# TAVILY_API_KEY=
# BRAVE_SEARCH_KEY=
# EXA_API_KEY=
SEARXNG_BASE_URL=http://localhost:8888

# Server Configuration
PORT=8000
HOST=0.0.0.0
DEBUG=True

# Storage Paths
DB_PATH=agents.db
CHROMA_PATH=.\chroma_db
TRACE_DIR=.\logs

# Agent Configuration
MAX_TOOL_TURNS=5
DEFAULT_MODEL=llama3.2
EMBED_MODEL=nomic-embed-text

# Voice (Optional)
# DEEPGRAM_API_KEY=
# GOOGLE_PLACES_API_KEY=
"@
        
        Set-Content -Path $envFile -Value $envContent
        Write-Success ".env file created with defaults"
    }
    
    Write-Host ""
    Write-Warning-Custom "⚠️  Please update the .env file with your API keys:"
    Write-Host "   - Open: $envFile in your editor"
    Write-Host ""
    Write-Host "   Required: At least one LLM API key (Groq, OpenAI, etc.) OR local Ollama instance"
    Write-Host "   Optional: Search API keys, voice services, etc."
    Write-Host ""
    
    return $true
}

function New-DataDirectories {
    Write-Header "Creating Data Directories"
    
    # Note: agents.db and other SQLite files are created by the backend on first run
    $dirs = @("chroma_db", "logs", "data")
    
    foreach ($dir in $dirs) {
        $dirPath = Join-Path $ScriptDir $dir
        if (-not (Test-Path $dirPath)) {
            New-Item -ItemType Directory -Path $dirPath -Force | Out-Null
            Write-Success "Created directory: $dir\"
        }
    }
}

################################################################################
# OPTIONAL SERVICES
################################################################################

function Show-OptionalServices {
    Write-Header "Optional Services"
    
    Write-Host "The following services are OPTIONAL but recommended:" -ForegroundColor Cyan
    Write-Host ""
    
    # Ollama
    Write-Host "1. Ollama (Local LLM Inference)" -ForegroundColor Yellow
    if (Test-CommandExists ollama) {
        Write-Success "Ollama found"
    }
    else {
        Write-Host "   Install from: https://ollama.ai"
        Write-Host "   Then pull a model: ollama pull llama3.2"
    }
    Write-Host ""
    
    # Docker
    Write-Host "2. Docker (for SearXNG Search & Sandboxing)" -ForegroundColor Yellow
    if (Test-CommandExists docker) {
        Write-Success "Docker found: Desktop running or CLI available"
    }
    else {
        Write-Host "   Install from: https://www.docker.com/products/docker-desktop"
        Write-Host "   Needed for: SearXNG search backend, isolated code execution"
    }
    Write-Host ""
    
    # Git
    Write-Host "3. Git (Version Control)" -ForegroundColor Yellow
    if (Test-CommandExists git) {
        Write-Success "Git found"
    }
    else {
        Write-Host "   Install from: https://git-scm.com"
    }
    Write-Host ""
}

function Setup-DockerServices {
    Write-Header "Setting Up Docker Services (Optional)"
    
    if (-not (Test-CommandExists docker)) {
        Write-Warning-Custom "Docker not installed - skipping optional services"
        return $true
    }
    
    if (-not (Test-CommandExists docker-compose)) {
        Write-Warning-Custom "Docker Compose not installed - skipping optional services"
        return $true
    }
    
    if ($SkipDocker) {
        Write-Info "Skipping Docker services setup (-SkipDocker flag)"
        return $true
    }
    
    $response = Read-Host "Start Docker services (SearXNG)? (y/n)"
    
    if ($response -eq "y") {
        Write-Info "Starting Docker services via docker-compose..."
        
        try {
            & docker-compose up -d
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Docker services started"
                Write-Host ""
                Write-Info "Service URLs:"
                Write-Host "  SearXNG: http://localhost:8888"
            }
            else {
                Write-Warning-Custom "Failed to start Docker services"
            }
        }
        catch {
            Write-Warning-Custom "Error starting Docker services: $_"
        }
    }
    
    return $true
}

################################################################################
# GETTING STARTED GUIDE
################################################################################

function Show-GettingStarted {
    Write-Header "✓ Setup Complete!"
    
    Write-Host ""
    Write-Host "Your Agent Platform is ready to run!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📝 Next Steps:" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "1. " -NoNewline
    Write-Host "Configure your environment:" -ForegroundColor Yellow
    Write-Host "   Edit: $ScriptDir\.env"
    Write-Host "   Add API keys for your chosen LLM provider"
    Write-Host ""
    
    Write-Host "2. " -NoNewline
    Write-Host "Start local LLM (optional but recommended):" -ForegroundColor Yellow
    Write-Host "   ollama run llama3.2"
    Write-Host ""
    
    Write-Host "3. " -NoNewline
    Write-Host "Start services in separate PowerShell terminals:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Terminal 1 - Backend:"
    Write-Host "   .\venv\Scripts\Activate.ps1"
    Write-Host "   npm run dev:backend"
    Write-Host ""
    Write-Host "   Terminal 2 - Frontend:"
    Write-Host "   npm run dev:frontend"
    Write-Host ""
    Write-Host "   Terminal 3 - Docker services (optional):"
    Write-Host "   npm run dev:services"
    Write-Host ""
    
    Write-Host "4. " -NoNewline
    Write-Host "Or run everything at once:" -ForegroundColor Yellow
    Write-Host "   npm run dev"
    Write-Host ""
    
    Write-Host "5. " -NoNewline
    Write-Host "Open in browser:" -ForegroundColor Yellow
    Write-Host "   http://localhost:5173"
    Write-Host ""
    
    Write-Host "📚 Documentation:" -ForegroundColor Cyan
    Write-Host "   - README.md - Project overview"
    Write-Host "   - DEVELOPER_GUIDE.md - Architecture details"
    Write-Host "   - FEATURES_AND_ROADMAP.md - Planned features"
    Write-Host ""
    
    Write-Host "🤝 Support:" -ForegroundColor Cyan
    Write-Host "   - Check existing issues on GitHub"
    Write-Host "   - Review CONTRIBUTING.md for guidelines"
    Write-Host ""
    
    Write-Host "💡 Tips:" -ForegroundColor Cyan
    Write-Host "   - Use 'npm run' to see all available commands"
    Write-Host "   - Logs are saved in .\logs directory"
    Write-Host "   - Session data goes to .\chroma_db"
    Write-Host ""
}

################################################################################
# MAIN EXECUTION
################################################################################

function Main {
    Clear-Host
    Write-Header "$ProjectName - Fresh Installation Setup"
    
    Write-Host "This script will set up the Agent Platform on your Windows system." -ForegroundColor White
    Write-Host "Estimated time: 5-15 minutes (depending on internet speed)" -ForegroundColor White
    Write-Host ""
    
    # Check prerequisites
    if (-not $SkipPrerequisiteCheck) {
        if (-not (Test-Prerequisites)) {
            Write-Error-Custom "Please install missing dependencies and try again"
            exit 1
        }
    }
    
    Write-Host ""
    
    # Setup Python
    if (-not (Initialize-PythonEnvironment)) {
        exit 1
    }
    
    Write-Host ""
    
    # Install dependencies
    if (-not (Install-PythonDependencies)) {
        exit 1
    }
    
    Write-Host ""
    
    if (-not (Install-NodeDependencies)) {
        exit 1
    }
    
    Write-Host ""
    
    # Setup configuration
    if (-not (Initialize-EnvironmentFile)) {
        exit 1
    }
    
    Write-Host ""
    
    New-DataDirectories
    
    Write-Host ""
    
    # Optional services info
    Show-OptionalServices
    
    Write-Host ""
    
    # Optional: Setup Docker
    if (-not $SkipDocker) {
        Setup-DockerServices
    }
    
    Write-Host ""
    
    # Show getting started guide
    Show-GettingStarted
}

# Run main function
Main
