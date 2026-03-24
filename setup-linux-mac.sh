#!/bin/bash

################################################################################
# Agent Platform Setup Script for macOS & Linux
# Version: 1.0
# Supports: Ubuntu 20.04+, macOS 12+, and other Unix-like systems
################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="Agent Platform"
PYTHON_MIN_VERSION=3.10
NODE_MIN_VERSION=18

################################################################################
# UTILITY FUNCTIONS
################################################################################

print_header() {
    echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ $1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

get_version() {
    local cmd=$1
    local version_flag=${2:-"--version"}
    
    if check_command "$cmd"; then
        $cmd $version_flag 2>/dev/null | head -n 1 || echo "unknown"
    else
        echo "not installed"
    fi
}

compare_versions() {
    printf '%s\n' "$1" "$2" | sort -V | head -n 1
}

################################################################################
# PREREQUISITE CHECKS
################################################################################

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    local missing_deps=0
    
    # Check Python
    if check_command python3; then
        local py_version=$(python3 --version 2>&1 | awk '{print $2}')
        local py_major_minor=$(echo "$py_version" | cut -d. -f1,2)
        
        if [ "$(compare_versions "$PYTHON_MIN_VERSION" "$py_major_minor")" = "$PYTHON_MIN_VERSION" ]; then
            print_success "Python $py_version installed"
        else
            print_error "Python $py_version installed (requires $PYTHON_MIN_VERSION+)"
            missing_deps=1
        fi
    else
        print_error "Python 3 not found"
        missing_deps=1
    fi
    
    # Check Node.js
    if check_command node; then
        local node_version=$(node --version | sed 's/v//')
        local node_major=$(echo "$node_version" | cut -d. -f1)
        
        if [ "$node_major" -ge "$NODE_MIN_VERSION" ]; then
            print_success "Node.js $node_version installed"
        else
            print_error "Node.js $node_version installed (requires $NODE_MIN_VERSION+)"
            missing_deps=1
        fi
    else
        print_error "Node.js not found"
        missing_deps=1
    fi
    
    # Check Git (recommended)
    if check_command git; then
        print_success "Git $(git --version | awk '{print $3}') installed"
    else
        print_warning "Git not found (recommended for version control)"
    fi
    
    if [ $missing_deps -eq 1 ]; then
        print_error "Missing required dependencies"
        echo ""
        echo "Installation instructions for your system:"
        echo ""
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "📦 macOS - Install via Homebrew:"
            echo "  brew install python@3.11 node git"
        else
            echo "📦 Ubuntu/Debian - Install via apt:"
            echo "  sudo apt update"
            echo "  sudo apt install -y python3.11 python3.11-venv python3-pip nodejs npm git"
            echo ""
            echo "📦 Fedora/RHEL - Install via dnf:"
            echo "  sudo dnf install python3.11 python3-pip nodejs npm git"
        fi
        return 1
    fi
    
    return 0
}

################################################################################
# SYSTEM SETUP
################################################################################

setup_python_environment() {
    print_header "Setting Up Python Environment"
    
    local venv_path="${SCRIPT_DIR}/venv"
    
    if [ -d "$venv_path" ]; then
        print_warning "Virtual environment already exists at $venv_path"
        read -p "Recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$venv_path"
            print_info "Removing old virtual environment..."
        else
            print_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    print_info "Creating virtual environment..."
    python3 -m venv "$venv_path"
    
    # Activate venv
    source "${venv_path}/bin/activate"
    
    print_success "Virtual environment created and activated"
    
    # Upgrade pip
    print_info "Upgrading pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel
    
    print_success "Python environment ready"
}

install_python_dependencies() {
    print_header "Installing Python Dependencies"
    
    local requirements_file="${SCRIPT_DIR}/requirements.txt"
    
    if [ ! -f "$requirements_file" ]; then
        print_error "requirements.txt not found at $requirements_file"
        return 1
    fi
    
    print_info "Installing packages from requirements.txt..."
    pip install -r "$requirements_file"
    
    print_success "Python dependencies installed"
}

install_node_dependencies() {
    print_header "Installing Node.js Dependencies"
    
    print_info "Installing root dependencies..."
    npm install
    
    print_info "Installing frontend dependencies..."
    cd "${SCRIPT_DIR}/frontend"
    npm install
    cd "${SCRIPT_DIR}"
    
    print_success "Node.js dependencies installed"
}

################################################################################
# CONFIGURATION SETUP
################################################################################

setup_environment_file() {
    print_header "Setting Up Environment Configuration"
    
    local env_file="${SCRIPT_DIR}/.env"
    local env_example="${SCRIPT_DIR}/.env.example"
    
    if [ -f "$env_file" ]; then
        print_warning ".env file already exists"
        read -p "Overwrite with template? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Keeping existing .env file"
            return 0
        fi
    fi
    
    if [ -f "$env_example" ]; then
        cp "$env_example" "$env_file"
        print_success ".env file created from .env.example"
    else
        # Create a default .env file
        cat > "$env_file" << 'EOF'
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
CHROMA_PATH=./chroma_db
TRACE_DIR=./logs

# Agent Configuration
MAX_TOOL_TURNS=5
DEFAULT_MODEL=llama3.2
EMBED_MODEL=nomic-embed-text

# Voice (Optional)
# DEEPGRAM_API_KEY=
# GOOGLE_PLACES_API_KEY=
EOF
        print_success ".env file created with defaults"
    fi
    
    echo ""
    print_warning "⚠️  Please update the .env file with your API keys:"
    echo "   - nano .env  (or your preferred editor)"
    echo ""
    echo "   Required: At least one LLM API key (Groq, OpenAI, etc.) OR local Ollama instance"
    echo "   Optional: Search API keys, voice services, etc."
    echo ""
}

setup_data_directories() {
    print_header "Creating Data Directories"
    
    # Note: agents.db and other SQLite files are created by the backend on first run
    local dirs=("chroma_db" "logs" "data")
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "${SCRIPT_DIR}/${dir}" ]; then
            mkdir -p "${SCRIPT_DIR}/${dir}"
            print_success "Created directory: ${dir}/"
        fi
    done
}

################################################################################
# OPTIONAL SERVICES
################################################################################

check_optional_services() {
    print_header "Optional Services"
    
    echo "The following services are OPTIONAL but recommended:"
    echo ""
    
    # Ollama
    echo "${YELLOW}1. Ollama (Local LLM Inference)${NC}"
    if check_command ollama; then
        print_success "Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
    else
        echo "   Install from: ${BLUE}https://ollama.ai${NC}"
        echo "   Then pull a model: ${BLUE}ollama pull llama3.2${NC}"
    fi
    echo ""
    
    # Docker
    echo "${YELLOW}2. Docker (for SearXNG Search & Sandboxing)${NC}"
    if check_command docker; then
        print_success "Docker found: $(docker --version)"
    else
        echo "   Install from: ${BLUE}https://www.docker.com/products/docker-desktop${NC}"
        echo "   Needed for: SearXNG search backend, isolated code execution"
    fi
    echo ""
    
    # Git
    echo "${YELLOW}3. Git (Version Control)${NC}"
    if check_command git; then
        print_success "Git found: $(git --version)"
    else
        echo "   Install from: ${BLUE}https://git-scm.com${NC}"
    fi
    echo ""
}

setup_docker_services() {
    print_header "Setting Up Docker Services (Optional)"
    
    if ! check_command docker; then
        print_warning "Docker not installed - skipping optional services"
        return 0
    fi
    
    if ! check_command docker-compose; then
        print_warning "Docker Compose not installed - skipping optional services"
        return 0
    fi
    
    read -p "Start Docker services (SearXNG)? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Starting Docker services via docker-compose..."
        docker-compose up -d
        print_success "Docker services started"
        
        echo ""
        print_info "Service URLs:"
        echo "  SearXNG: ${BLUE}http://localhost:8888${NC}"
    fi
}

################################################################################
# POST-SETUP GUIDANCE
################################################################################

print_getting_started() {
    print_header "✓ Setup Complete!"
    
    cat << EOF

${GREEN}Your Agent Platform is ready to run!${NC}

${BLUE}📝 Next Steps:${NC}

1. ${YELLOW}Configure your environment:${NC}
   - Edit: ${BLUE}.env${NC}
   - Add API keys for your chosen LLM provider

2. ${YELLOW}Start local LLM (optional but recommended):${NC}
   ${BLUE}ollama run llama3.2${NC}

3. ${YELLOW}Start services in separate terminals:${NC}

   Terminal 1 - Backend:
   ${BLUE}source venv/bin/activate${NC}
   ${BLUE}npm run dev:backend${NC}

   Terminal 2 - Frontend:
   ${BLUE}npm run dev:frontend${NC}

   Terminal 3 - Docker services (optional):
   ${BLUE}npm run dev:services${NC}

4. ${YELLOW}Or run everything at once:${NC}
   ${BLUE}npm run dev${NC}

5. ${YELLOW}Open in browser:${NC}
   ${BLUE}http://localhost:5173${NC}

${BLUE}📚 Documentation:${NC}
   - README.md - Project overview
   - DEVELOPER_GUIDE.md - Architecture details
   - FEATURES_AND_ROADMAP.md - Planned features

${BLUE}🤝 Support:${NC}
   - Check existing issues on GitHub
   - Review CONTRIBUTING.md for guidelines

${BLUE}💡 Tips:${NC}
   - Use 'npm run' to see all available commands
   - Logs are saved in ./logs directory
   - Session data goes to ./chroma_db

EOF
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    clear
    print_header "Agent Platform - Fresh Installation Setup"
    
    echo "This script will set up the Agent Platform on your system."
    echo "Estimated time: 5-15 minutes (depending on internet speed)"
    echo ""
    
    # Check prerequisites
    if ! check_prerequisites; then
        print_error "Please install missing dependencies and try again"
        exit 1
    fi
    echo ""
    
    # Setup Python
    setup_python_environment
    echo ""
    
    # Install dependencies
    install_python_dependencies
    echo ""
    
    install_node_dependencies
    echo ""
    
    # Setup configuration
    setup_environment_file
    echo ""
    
    setup_data_directories
    echo ""
    
    # Optional services info
    check_optional_services
    echo ""
    
    # Optional: Setup Docker
    setup_docker_services
    echo ""
    
    # Print getting started guide
    print_getting_started
}

# Run main function
main "$@"
