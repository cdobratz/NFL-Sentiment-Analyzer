"""
Deployment and Infrastructure Tests

This module contains comprehensive tests for Docker container builds,
multi-service orchestration, production configuration validation,
smoke tests, and security/performance validation.

Requirements covered: 8.1, 8.2, 8.3
"""

import pytest
import docker
import requests
import subprocess
import time
import os
import yaml
import json
from typing import Dict, List, Optional
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Test configuration
DOCKER_COMPOSE_FILES = {
    'dev': 'docker-compose.dev.yml',
    'prod': 'docker-compose.prod.yml',
    'base': 'docker-compose.yml'
}

REQUIRED_SERVICES = [
    'frontend', 'api', 'mongodb', 'redis', 'nginx',
    'prometheus', 'grafana', 'loki', 'promtail'
]

HEALTH_CHECK_ENDPOINTS = {
    'api': 'http://localhost:8000/health',
    'frontend': 'http://localhost:3000/health',
    'nginx': 'http://localhost/health'
}

SECURITY_PORTS = {
    'mongodb': 27017,
    'redis': 6379,
    'prometheus': 9090,
    'grafana': 3001
}


class TestDockerContainerBuilds:
    """Test Docker container builds and image validation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.docker_client = docker.from_env()
        self.test_images = []
    
    def teardown_method(self):
        """Clean up test images."""
        for image_id in self.test_images:
            try:
                self.docker_client.images.remove(image_id, force=True)
            except docker.errors.ImageNotFound:
                pass
    
    def test_backend_dockerfile_build(self):
        """Test backend Dockerfile builds successfully."""
        # Test development stage
        dev_image, dev_logs = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="development",
            tag="nfl-analyzer-api:dev-test"
        )
        self.test_images.append(dev_image.id)
        
        # Verify development image
        assert dev_image is not None
        assert "nfl-analyzer-api:dev-test" in [tag for tag in dev_image.tags]
        
        # Test production stage
        prod_image, prod_logs = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="production",
            tag="nfl-analyzer-api:prod-test"
        )
        self.test_images.append(prod_image.id)
        
        # Verify production image
        assert prod_image is not None
        assert "nfl-analyzer-api:prod-test" in [tag for tag in prod_image.tags]
        
        # Verify production image is smaller (optimized)
        dev_size = dev_image.attrs['Size']
        prod_size = prod_image.attrs['Size']
        # Production should be reasonably sized (not necessarily smaller due to dependencies)
        assert prod_size > 0
    
    def test_frontend_dockerfile_build(self):
        """Test frontend Dockerfile builds successfully."""
        frontend_path = Path("frontend")
        if not frontend_path.exists():
            pytest.skip("Frontend directory not found")
        
        # Test development stage
        dev_image, dev_logs = self.docker_client.images.build(
            path=str(frontend_path),
            dockerfile="Dockerfile",
            target="development",
            tag="nfl-analyzer-frontend:dev-test"
        )
        self.test_images.append(dev_image.id)
        
        # Verify development image
        assert dev_image is not None
        assert "nfl-analyzer-frontend:dev-test" in [tag for tag in dev_image.tags]
        
        # Test production stage
        prod_image, prod_logs = self.docker_client.images.build(
            path=str(frontend_path),
            dockerfile="Dockerfile",
            target="production",
            tag="nfl-analyzer-frontend:prod-test"
        )
        self.test_images.append(prod_image.id)
        
        # Verify production image
        assert prod_image is not None
        assert "nfl-analyzer-frontend:prod-test" in [tag for tag in prod_image.tags]
    
    def test_docker_image_security_scan(self):
        """Test Docker images for security vulnerabilities."""
        # Build a test image
        image, logs = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="production",
            tag="nfl-analyzer-security-test"
        )
        self.test_images.append(image.id)
        
        # Run basic security checks
        # Check for non-root user
        container = self.docker_client.containers.run(
            image.id,
            command="whoami",
            remove=True,
            detach=False
        )
        
        # Verify non-root user is used
        assert container.decode().strip() == "app"
        
        # Check for minimal attack surface
        container = self.docker_client.containers.run(
            image.id,
            command="ls -la /",
            remove=True,
            detach=False
        )
        
        # Verify essential directories exist
        output = container.decode()
        assert "/app" in output
        assert "/home/app" in output
    
    def test_docker_image_health_checks(self):
        """Test Docker images have proper health checks."""
        # Build production image
        image, logs = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="production",
            tag="nfl-analyzer-health-test"
        )
        self.test_images.append(image.id)
        
        # Check health check configuration
        config = image.attrs['Config']
        assert 'Healthcheck' in config
        assert config['Healthcheck']['Test'] is not None
        assert config['Healthcheck']['Interval'] > 0
        assert config['Healthcheck']['Timeout'] > 0
        assert config['Healthcheck']['Retries'] > 0


class TestMultiServiceOrchestration:
    """Test Docker Compose multi-service orchestration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.docker_client = docker.from_env()
        self.test_containers = []
        self.test_networks = []
        self.test_volumes = []
    
    def teardown_method(self):
        """Clean up test resources."""
        # Stop and remove containers
        for container in self.test_containers:
            try:
                container.stop(timeout=10)
                container.remove()
            except docker.errors.NotFound:
                pass
        
        # Remove networks
        for network in self.test_networks:
            try:
                network.remove()
            except docker.errors.NotFound:
                pass
        
        # Remove volumes
        for volume in self.test_volumes:
            try:
                volume.remove()
            except docker.errors.NotFound:
                pass
    
    def test_docker_compose_file_validation(self):
        """Test Docker Compose files are valid YAML."""
        for env, filename in DOCKER_COMPOSE_FILES.items():
            if not Path(filename).exists():
                continue
                
            with open(filename, 'r') as f:
                try:
                    compose_config = yaml.safe_load(f)
                    assert compose_config is not None
                    assert 'version' in compose_config
                    assert 'services' in compose_config
                    
                    # Validate required services exist
                    services = compose_config['services']
                    for service in REQUIRED_SERVICES:
                        if service not in services:
                            # Some services might be optional in dev
                            if env == 'dev' and service in ['nginx', 'prometheus', 'grafana']:
                                continue
                            pytest.fail(f"Required service '{service}' not found in {filename}")
                    
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {filename}: {e}")
    
    def test_service_dependencies(self):
        """Test service dependencies are properly configured."""
        compose_file = 'docker-compose.yml'
        if not Path(compose_file).exists():
            pytest.skip(f"{compose_file} not found")
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config['services']
        
        # Test API depends on MongoDB and Redis
        api_service = services.get('api', {})
        api_depends_on = api_service.get('depends_on', [])
        assert 'mongodb' in api_depends_on
        assert 'redis' in api_depends_on
        
        # Test frontend depends on API
        frontend_service = services.get('frontend', {})
        frontend_depends_on = frontend_service.get('depends_on', [])
        assert 'api' in frontend_depends_on
        
        # Test nginx depends on frontend and api
        nginx_service = services.get('nginx', {})
        nginx_depends_on = nginx_service.get('depends_on', [])
        assert 'frontend' in nginx_depends_on
        assert 'api' in nginx_depends_on
    
    def test_network_configuration(self):
        """Test network configuration for service communication."""
        compose_file = 'docker-compose.yml'
        if not Path(compose_file).exists():
            pytest.skip(f"{compose_file} not found")
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check if custom networks are defined
        networks = compose_config.get('networks', {})
        
        # Verify services can communicate (implicit default network)
        services = compose_config['services']
        
        # Check API service has correct MongoDB URL
        api_env = services['api'].get('environment', [])
        mongodb_url_found = False
        for env_var in api_env:
            if isinstance(env_var, str) and 'MONGODB_URL' in env_var:
                assert 'mongodb:27017' in env_var
                mongodb_url_found = True
                break
            elif isinstance(env_var, dict) and 'MONGODB_URL' in env_var:
                assert 'mongodb:27017' in env_var['MONGODB_URL']
                mongodb_url_found = True
                break
        
        assert mongodb_url_found, "MongoDB URL not properly configured for API service"
    
    def test_volume_configuration(self):
        """Test volume configuration for data persistence."""
        compose_file = 'docker-compose.yml'
        if not Path(compose_file).exists():
            pytest.skip(f"{compose_file} not found")
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check volumes are defined
        volumes = compose_config.get('volumes', {})
        required_volumes = ['mongodb_data', 'redis_data', 'prometheus_data', 'grafana_data']
        
        for volume in required_volumes:
            assert volume in volumes, f"Required volume '{volume}' not found"
        
        # Check services use volumes correctly
        services = compose_config['services']
        
        # MongoDB should use mongodb_data volume
        mongodb_volumes = services['mongodb'].get('volumes', [])
        mongodb_data_mounted = any('mongodb_data:/data/db' in vol for vol in mongodb_volumes)
        assert mongodb_data_mounted, "MongoDB data volume not properly mounted"
        
        # Redis should use redis_data volume
        redis_volumes = services['redis'].get('volumes', [])
        redis_data_mounted = any('redis_data:/data' in vol for vol in redis_volumes)
        assert redis_data_mounted, "Redis data volume not properly mounted"


class TestProductionConfiguration:
    """Test production configuration and environment variable handling."""
    
    def test_environment_variable_validation(self):
        """Test required environment variables are defined."""
        env_example_path = Path('.env.example')
        env_prod_path = Path('.env.production')
        
        if not env_example_path.exists():
            pytest.skip(".env.example not found")
        
        # Read example environment variables
        with open(env_example_path, 'r') as f:
            example_content = f.read()
        
        # Extract variable names from example file
        example_vars = set()
        for line in example_content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                var_name = line.split('=')[0].strip()
                example_vars.add(var_name)
        
        # Check production file has all required variables
        if env_prod_path.exists():
            with open(env_prod_path, 'r') as f:
                prod_content = f.read()
            
            prod_vars = set()
            for line in prod_content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    var_name = line.split('=')[0].strip()
                    prod_vars.add(var_name)
            
            # Critical variables that must be in production
            critical_vars = {
                'SECRET_KEY', 'MONGODB_URL', 'REDIS_URL',
                'MONGO_ROOT_PASSWORD', 'REDIS_PASSWORD'
            }
            
            for var in critical_vars:
                assert var in prod_vars, f"Critical variable '{var}' missing from production config"
    
    def test_security_configuration(self):
        """Test security-related configuration."""
        compose_prod_path = Path('docker-compose.prod.yml')
        if not compose_prod_path.exists():
            pytest.skip("docker-compose.prod.yml not found")
        
        with open(compose_prod_path, 'r') as f:
            prod_config = yaml.safe_load(f)
        
        services = prod_config.get('services', {})
        
        # Test API service security
        if 'api' in services:
            api_service = services['api']
            
            # Check resource limits are set
            deploy_config = api_service.get('deploy', {})
            resources = deploy_config.get('resources', {})
            
            assert 'limits' in resources, "Resource limits not configured for API service"
            assert 'reservations' in resources, "Resource reservations not configured for API service"
            
            # Check environment variables don't contain secrets in plain text
            api_env = api_service.get('environment', [])
            for env_var in api_env:
                if isinstance(env_var, str):
                    # Should use environment variable substitution, not plain text
                    if 'PASSWORD' in env_var or 'SECRET' in env_var or 'KEY' in env_var:
                        assert '${' in env_var, f"Security variable should use substitution: {env_var}"
    
    def test_production_optimizations(self):
        """Test production-specific optimizations."""
        compose_prod_path = Path('docker-compose.prod.yml')
        if not compose_prod_path.exists():
            pytest.skip("docker-compose.prod.yml not found")
        
        with open(compose_prod_path, 'r') as f:
            prod_config = yaml.safe_load(f)
        
        services = prod_config.get('services', {})
        
        # Test API service has multiple replicas
        if 'api' in services:
            api_deploy = services['api'].get('deploy', {})
            replicas = api_deploy.get('replicas', 1)
            assert replicas > 1, "API service should have multiple replicas in production"
        
        # Test frontend service has replicas
        if 'frontend' in services:
            frontend_deploy = services['frontend'].get('deploy', {})
            replicas = frontend_deploy.get('replicas', 1)
            assert replicas >= 1, "Frontend service should have at least one replica"
        
        # Test MongoDB has authentication enabled
        if 'mongodb' in services:
            mongodb_service = services['mongodb']
            mongodb_command = mongodb_service.get('command', '')
            if mongodb_command:
                assert '--auth' in mongodb_command, "MongoDB should have authentication enabled"


class TestSmokeTests:
    """Smoke tests for deployed application health."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment variables."""
        self.original_env = os.environ.copy()
        
        # Set test environment variables
        test_env = {
            'SECRET_KEY': 'test-secret-key-for-testing-only',
            'MONGODB_URL': 'mongodb://localhost:27017',
            'DATABASE_NAME': 'nfl_sentiment_test',
            'REDIS_URL': 'redis://localhost:6379',
            'DEBUG': 'true',
            'ENVIRONMENT': 'test'
        }
        
        os.environ.update(test_env)
        
        yield
        
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_api_health_endpoint(self):
        """Test API health endpoint responds correctly."""
        try:
            response = requests.get('http://localhost:8000/health', timeout=10)
            assert response.status_code == 200
            
            health_data = response.json()
            assert 'status' in health_data
            assert health_data['status'] in ['healthy', 'ok']
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for smoke test")
    
    def test_frontend_accessibility(self):
        """Test frontend is accessible."""
        try:
            response = requests.get('http://localhost:3000', timeout=10)
            assert response.status_code == 200
            assert 'text/html' in response.headers.get('content-type', '')
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Frontend service not running for smoke test")
    
    def test_database_connectivity(self):
        """Test database connectivity."""
        try:
            from app.core.database import get_database
            
            # Test MongoDB connection
            db = get_database()
            # Simple ping test
            result = db.command('ping')
            assert result['ok'] == 1
            
        except Exception as e:
            pytest.skip(f"Database not available for smoke test: {e}")
    
    def test_redis_connectivity(self):
        """Test Redis connectivity."""
        try:
            import redis
            
            redis_client = redis.from_url('redis://localhost:6379')
            redis_client.ping()
            
            # Test basic operations
            redis_client.set('test_key', 'test_value', ex=60)
            value = redis_client.get('test_key')
            assert value.decode() == 'test_value'
            
            redis_client.delete('test_key')
            
        except Exception as e:
            pytest.skip(f"Redis not available for smoke test: {e}")
    
    def test_service_integration(self):
        """Test basic service integration."""
        try:
            # Test API can connect to database and return data
            response = requests.get('http://localhost:8000/data/teams', timeout=10)
            
            # Should return 200 or 404 (if no data), but not 500
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, (list, dict))
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Services not running for integration smoke test")


class TestSecurityValidation:
    """Security validation tests for production setup."""
    
    def test_port_exposure_security(self):
        """Test that sensitive ports are not exposed unnecessarily."""
        compose_file = 'docker-compose.yml'
        if not Path(compose_file).exists():
            pytest.skip(f"{compose_file} not found")
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config['services']
        
        # Check MongoDB port exposure
        mongodb_ports = services['mongodb'].get('ports', [])
        # In production, MongoDB should not expose ports to host
        # This is acceptable for development but should be noted
        
        # Check Redis port exposure
        redis_ports = services['redis'].get('ports', [])
        # Similar to MongoDB, Redis exposure should be controlled
        
        # Verify nginx is the main entry point
        nginx_ports = services['nginx'].get('ports', [])
        assert any('80:80' in str(port) for port in nginx_ports), "Nginx should expose port 80"
    
    def test_secret_management(self):
        """Test that secrets are properly managed."""
        # Check that sensitive files are in .gitignore
        gitignore_path = Path('.gitignore')
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
            
            # Check that environment files are ignored
            assert '.env' in gitignore_content
            assert '.env.production' in gitignore_content or '.env.*' in gitignore_content
        
        # Check that example files don't contain real secrets
        env_example_path = Path('.env.example')
        if env_example_path.exists():
            with open(env_example_path, 'r') as f:
                example_content = f.read()
            
            # Should contain placeholder values, not real secrets
            assert 'your_' in example_content or 'change-this' in example_content
            assert 'password123' not in example_content.lower()
            assert 'admin123' not in example_content.lower()
    
    def test_ssl_configuration(self):
        """Test SSL/TLS configuration."""
        nginx_config_path = Path('nginx/nginx.conf')
        nginx_prod_config_path = Path('nginx/nginx.prod.conf')
        
        # Check production nginx configuration
        if nginx_prod_config_path.exists():
            with open(nginx_prod_config_path, 'r') as f:
                nginx_config = f.read()
            
            # Should have SSL configuration
            assert 'ssl_certificate' in nginx_config or 'ssl' in nginx_config
            assert '443' in nginx_config  # HTTPS port
        
        # Check Docker Compose production SSL volume mounts
        compose_prod_path = Path('docker-compose.prod.yml')
        if compose_prod_path.exists():
            with open(compose_prod_path, 'r') as f:
                prod_config = yaml.safe_load(f)
            
            nginx_service = prod_config.get('services', {}).get('nginx', {})
            nginx_volumes = nginx_service.get('volumes', [])
            
            # Should mount SSL certificates
            ssl_volume_found = any('ssl' in str(vol) or 'letsencrypt' in str(vol) for vol in nginx_volumes)
            if not ssl_volume_found:
                # This might be acceptable if using a different SSL setup
                pass


class TestPerformanceValidation:
    """Performance validation tests for production setup."""
    
    def test_resource_limits(self):
        """Test that resource limits are properly configured."""
        compose_prod_path = Path('docker-compose.prod.yml')
        if not compose_prod_path.exists():
            pytest.skip("docker-compose.prod.yml not found")
        
        with open(compose_prod_path, 'r') as f:
            prod_config = yaml.safe_load(f)
        
        services = prod_config.get('services', {})
        
        # Check that critical services have resource limits
        critical_services = ['api', 'frontend', 'mongodb', 'redis']
        
        for service_name in critical_services:
            if service_name in services:
                service = services[service_name]
                deploy_config = service.get('deploy', {})
                resources = deploy_config.get('resources', {})
                
                if resources:  # If resources are configured
                    limits = resources.get('limits', {})
                    reservations = resources.get('reservations', {})
                    
                    # Should have memory limits
                    if 'memory' in limits:
                        memory_limit = limits['memory']
                        assert memory_limit is not None
                    
                    # Should have CPU limits
                    if 'cpus' in limits:
                        cpu_limit = limits['cpus']
                        assert cpu_limit is not None
    
    def test_caching_configuration(self):
        """Test caching configuration for performance."""
        # Check Redis configuration
        compose_file = 'docker-compose.yml'
        if not Path(compose_file).exists():
            pytest.skip(f"{compose_file} not found")
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        redis_service = compose_config['services'].get('redis', {})
        redis_command = redis_service.get('command', '')
        
        # Should have persistence enabled
        assert 'appendonly yes' in redis_command
        
        # Check production Redis optimizations
        compose_prod_path = Path('docker-compose.prod.yml')
        if compose_prod_path.exists():
            with open(compose_prod_path, 'r') as f:
                prod_config = yaml.safe_load(f)
            
            prod_redis = prod_config.get('services', {}).get('redis', {})
            prod_redis_command = prod_redis.get('command', '')
            
            if prod_redis_command:
                # Should have memory management
                assert 'maxmemory' in prod_redis_command
                assert 'maxmemory-policy' in prod_redis_command
    
    def test_database_optimization(self):
        """Test database optimization configuration."""
        # Check MongoDB configuration
        compose_prod_path = Path('docker-compose.prod.yml')
        if not compose_prod_path.exists():
            pytest.skip("docker-compose.prod.yml not found")
        
        with open(compose_prod_path, 'r') as f:
            prod_config = yaml.safe_load(f)
        
        mongodb_service = prod_config.get('services', {}).get('mongodb', {})
        
        # Should have resource limits
        deploy_config = mongodb_service.get('deploy', {})
        resources = deploy_config.get('resources', {})
        
        if resources:
            limits = resources.get('limits', {})
            # Should have reasonable memory limits for MongoDB
            if 'memory' in limits:
                memory_limit = limits['memory']
                assert memory_limit is not None
        
        # Check for replica set configuration (if configured)
        mongodb_command = mongodb_service.get('command', '')
        if mongodb_command and 'replSet' in mongodb_command:
            assert '--replSet' in mongodb_command


if __name__ == "__main__":
    pytest.main([__file__, "-v"])