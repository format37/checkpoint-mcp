# source .env.prod
#!/bin/bash

# Source the environment file and export all variables
# set -a  # automatically export all variables
# source .env.prod
# set +a  # stop automatically exporting

echo "🐳 Starting Checkpoint MCP Server in PRODUCTION mode..."

# Check if mcp-shared network exists, create if not
if ! docker network ls | grep -q "mcp-shared"; then
    echo "📡 Creating mcp-shared network..."
    docker network create mcp-shared
fi

# Ensure production environment file exists
if [ ! -f ".env.prod" ]; then
    echo "❌ .env.prod file not found!"
    echo "Please copy .env.prod and configure it for production"
    exit 1
fi

# Ensure data directory exists with correct permissions
echo "📁 Setting up data directory..."
mkdir -p ./data
# Fix ownership to match the current user (who should match container UID 1000)
if [ -w ./data ]; then
    echo "✓ Data directory is writable"
else
    echo "⚠️  Fixing data directory permissions..."
    sudo chown -R $USER:$USER ./data
fi

# Stop and remove existing containers to ensure fresh deployment
echo "🛑 Stopping and removing existing containers..."
docker compose -f docker-compose.prod.yml down

# Remove existing images to force rebuild
echo "🗑️  Removing existing images to force rebuild..."
docker rmi -f checkpoint-mcp-checkpoint-mcp 2>/dev/null || true

# Build and start services in production mode
echo "🏗️  Building and starting services in production mode..."
docker compose -f docker-compose.prod.yml up --build -d

echo "✅ Checkpoint MCP Server is running in production mode!"
echo "🌐 Service is accessible via reverse proxy"
echo "🔧 MCP Endpoint: ${MCP_PUBLIC_BASE_URL}/checkpoint-mcp/"
echo ""
echo "To check logs:"
echo "  docker compose -f docker-compose.prod.yml logs -f"
echo ""
echo "To stop:"
echo "  docker compose -f docker-compose.prod.yml down"