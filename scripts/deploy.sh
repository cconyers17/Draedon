#!/bin/bash

# Deployment script for Text-to-CAD application on Render.com
# Supports staging and production deployments with health checks

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage function
usage() {
    echo "Usage: $0 [staging|production] [options]"
    echo ""
    echo "Environments:"
    echo "  staging     Deploy to staging environment"
    echo "  production  Deploy to production environment"
    echo ""
    echo "Options:"
    echo "  --skip-tests       Skip running tests before deployment"
    echo "  --skip-build       Skip building Docker images"
    echo "  --force            Force deployment without confirmations"
    echo "  --health-check     Run health checks after deployment"
    echo "  --rollback         Rollback to previous deployment"
    echo ""
    echo "Environment Variables:"
    echo "  RENDER_API_KEY     Render.com API key (required)"
    echo "  STAGING_SERVICE_ID Render staging service ID"
    echo "  PROD_SERVICE_ID    Render production service ID"
    exit 1
}

# Check requirements
check_requirements() {
    log_info "Checking deployment requirements..."

    # Check required commands
    local required_commands=("curl" "jq" "docker" "npm" "python3")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command '$cmd' not found"
            exit 1
        fi
    done

    # Check environment variables
    if [[ -z "$RENDER_API_KEY" ]]; then
        log_error "RENDER_API_KEY environment variable is not set"
        exit 1
    fi

    log_success "All requirements satisfied"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests (--skip-tests flag provided)"
        return 0
    fi

    log_info "Running tests..."

    # Frontend tests
    log_info "Running frontend tests..."
    cd "$PROJECT_ROOT/text-to-cad-app"
    npm ci
    npm run type-check
    npm run lint
    npm test -- --watchAll=false

    # Backend tests
    log_info "Running backend tests..."
    cd "$PROJECT_ROOT/backend"
    python3 -m pip install --upgrade pip
    pip install -r requirements.txt
    python3 -m pytest tests/ -v

    cd "$PROJECT_ROOT"
    log_success "All tests passed"
}

# Build Docker images
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_warning "Skipping build (--skip-build flag provided)"
        return 0
    fi

    log_info "Building Docker images..."

    # Build frontend image
    log_info "Building frontend image..."
    cd "$PROJECT_ROOT/text-to-cad-app"
    docker build -t text-to-cad-frontend:$DEPLOY_TIMESTAMP .

    # Build backend image
    log_info "Building backend image..."
    cd "$PROJECT_ROOT/backend"
    docker build -t text-to-cad-backend:$DEPLOY_TIMESTAMP .

    cd "$PROJECT_ROOT"
    log_success "Docker images built successfully"
}

# Deploy to Render
deploy_to_render() {
    local environment=$1
    local service_id=""

    if [[ "$environment" == "staging" ]]; then
        service_id="$STAGING_SERVICE_ID"
    elif [[ "$environment" == "production" ]]; then
        service_id="$PROD_SERVICE_ID"
    else
        log_error "Invalid environment: $environment"
        exit 1
    fi

    if [[ -z "$service_id" ]]; then
        log_error "${environment^^}_SERVICE_ID environment variable is not set"
        exit 1
    fi

    log_info "Deploying to $environment environment..."

    # Trigger deployment
    local deploy_response
    deploy_response=$(curl -s -X POST \
        -H "Accept: application/json" \
        -H "Authorization: Bearer $RENDER_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"clearCache": "clear"}' \
        "https://api.render.com/v1/services/$service_id/deploys")

    if [[ $? -ne 0 ]]; then
        log_error "Failed to trigger deployment"
        exit 1
    fi

    local deploy_id
    deploy_id=$(echo "$deploy_response" | jq -r '.id')

    if [[ "$deploy_id" == "null" ]]; then
        log_error "Failed to get deployment ID"
        echo "Response: $deploy_response"
        exit 1
    fi

    log_info "Deployment triggered with ID: $deploy_id"

    # Wait for deployment to complete
    wait_for_deployment "$service_id" "$deploy_id"

    log_success "Deployment to $environment completed successfully"
}

# Wait for deployment completion
wait_for_deployment() {
    local service_id=$1
    local deploy_id=$2
    local max_wait=1800  # 30 minutes
    local wait_time=0

    log_info "Waiting for deployment to complete..."

    while [[ $wait_time -lt $max_wait ]]; do
        local status
        status=$(curl -s \
            -H "Accept: application/json" \
            -H "Authorization: Bearer $RENDER_API_KEY" \
            "https://api.render.com/v1/services/$service_id/deploys/$deploy_id" | \
            jq -r '.status')

        case "$status" in
            "live")
                log_success "Deployment completed successfully"
                return 0
                ;;
            "build_failed"|"deploy_failed"|"canceled")
                log_error "Deployment failed with status: $status"
                exit 1
                ;;
            "in_progress"|"queued")
                echo -n "."
                sleep 30
                wait_time=$((wait_time + 30))
                ;;
            *)
                log_warning "Unknown deployment status: $status"
                sleep 30
                wait_time=$((wait_time + 30))
                ;;
        esac
    done

    log_error "Deployment timeout after ${max_wait}s"
    exit 1
}

# Health check
run_health_check() {
    local environment=$1
    local base_url=""

    if [[ "$environment" == "staging" ]]; then
        base_url="https://text-to-cad-staging.onrender.com"
    elif [[ "$environment" == "production" ]]; then
        base_url="https://text-to-cad.onrender.com"
    else
        log_error "Invalid environment for health check: $environment"
        return 1
    fi

    log_info "Running health checks for $environment..."

    # Wait for services to be ready
    sleep 60

    # Check backend health
    log_info "Checking backend health..."
    local backend_health
    backend_health=$(curl -s -f "$base_url/api/health" | jq -r '.status')

    if [[ "$backend_health" != "ok" ]]; then
        log_error "Backend health check failed"
        return 1
    fi

    # Check frontend health
    log_info "Checking frontend health..."
    local frontend_response
    frontend_response=$(curl -s -o /dev/null -w "%{http_code}" "$base_url/api/health")

    if [[ "$frontend_response" != "200" ]]; then
        log_error "Frontend health check failed (HTTP $frontend_response)"
        return 1
    fi

    log_success "Health checks passed for $environment"
}

# Rollback deployment
rollback_deployment() {
    local environment=$1
    log_warning "Rollback functionality not implemented yet"
    log_info "Please use Render.com dashboard to manually rollback if needed"
}

# Main deployment function
main() {
    local environment=""
    local skip_tests=false
    local skip_build=false
    local force=false
    local health_check=false
    local rollback=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            staging|production)
                environment=$1
                shift
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --skip-build)
                skip_build=true
                shift
                ;;
            --force)
                force=true
                shift
                ;;
            --health-check)
                health_check=true
                shift
                ;;
            --rollback)
                rollback=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done

    # Validate environment
    if [[ -z "$environment" ]]; then
        log_error "Environment not specified"
        usage
    fi

    # Set global variables
    SKIP_TESTS=$skip_tests
    SKIP_BUILD=$skip_build

    # Handle rollback
    if [[ "$rollback" == "true" ]]; then
        rollback_deployment "$environment"
        return 0
    fi

    # Confirmation for production
    if [[ "$environment" == "production" && "$force" != "true" ]]; then
        read -p "Are you sure you want to deploy to PRODUCTION? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log_info "Production deployment cancelled"
            exit 0
        fi
    fi

    log_info "Starting deployment to $environment environment"
    log_info "Timestamp: $DEPLOY_TIMESTAMP"

    # Run deployment steps
    check_requirements
    run_tests
    build_images
    deploy_to_render "$environment"

    # Optional health check
    if [[ "$health_check" == "true" ]]; then
        run_health_check "$environment"
    fi

    log_success "Deployment to $environment completed successfully!"
}

# Run main function with all arguments
main "$@"