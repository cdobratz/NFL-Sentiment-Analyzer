"""
Docker Build Tests

Specialized tests for Docker container builds, image optimization,
and multi-stage build validation.

Requirements covered: 8.1, 8.2
"""

import pytest
import docker
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional


class TestDockerBuildOptimization:
    """Test Docker build optimization and best practices."""
    
    def setup_method(self):
        """Set up test environment."""
        self.docker_client = docker.from_env()
        self.test_images = []
        self.temp_dirs = []
    
    def teardown_method(self):
        """Clean up test resources."""
        # Remove test images
        for image_id in self.test_images:
            try:
                self.docker_client.images.remove(image_id, force=True)
            except docker.errors.ImageNotFound:
                pass
        
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def test_dockerfile_best_practices(self):
        """Test Dockerfile follows best practices."""
        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            pytest.skip("Dockerfile not found")
        
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Check for multi-stage build
        assert "FROM python:3.11-slim as development" in dockerfile_content
        assert "FROM python:3.11-slim as production" in dockerfile_content
        
        # Check for non-root user
        assert "useradd" in dockerfile_content
        assert "USER app" in dockerfile_content
        
        # Check for health check
        assert "HEALTHCHECK" in dockerfile_content
        
        # Check for proper layer optimization
        assert "apt-get update && apt-get install" in dockerfile_content
        assert "rm -rf /var/lib/apt/lists/*" in dockerfile_content
        
        # Check for security updates
        assert "apt-get upgrade" in dockerfile_content
    
    def test_frontend_dockerfile_best_practices(self):
        """Test frontend Dockerfile follows best practices."""
        frontend_dockerfile = Path("frontend/Dockerfile")
        if not frontend_dockerfile.exists():
            pytest.skip("Frontend Dockerfile not found")
        
        with open(frontend_dockerfile, 'r') as f:
            dockerfile_content = f.read()
        
        # Should have multi-stage build
        stages = dockerfile_content.count("FROM ")
        assert stages >= 2, "Frontend should use multi-stage build"
        
        # Should use nginx for production
        assert "nginx" in dockerfile_content.lower()
        
        # Should copy built assets
        assert "COPY --from=" in dockerfile_content
    
    def test_build_context_optimization(self):
        """Test build context is optimized."""
        dockerignore_path = Path(".dockerignore")
        if not dockerignore_path.exists():
            pytest.skip(".dockerignore not found")
        
        with open(dockerignore_path, 'r') as f:
            dockerignore_content = f.read()
        
        # Should ignore common unnecessary files
        ignore_patterns = [
            "node_modules", ".git", "*.md", ".env*",
            "__pycache__", "*.pyc", ".pytest_cache",
            "tests", "docs"
        ]
        
        for pattern in ignore_patterns:
            assert pattern in dockerignore_content, f"Should ignore {pattern}"
    
    def test_image_layer_optimization(self):
        """Test Docker image layers are optimized."""
        # Build production image
        image, logs = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="production",
            tag="nfl-analyzer-layer-test"
        )
        self.test_images.append(image.id)
        
        # Get image history
        history = image.history()
        
        # Should have reasonable number of layers (not too many)
        assert len(history) < 20, "Too many layers in image"
        
        # Check for layer caching opportunities
        # RUN commands should be combined where possible
        dockerfile_path = Path("Dockerfile")
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Count RUN commands in production stage
        production_section = dockerfile_content.split("FROM python:3.11-slim as production")[1]
        run_count = production_section.count("RUN ")
        
        # Should have reasonable number of RUN commands
        assert run_count < 10, "Too many RUN commands, consider combining"
    
    def test_image_size_optimization(self):
        """Test Docker image size is optimized."""
        # Build both development and production images
        dev_image, _ = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="development",
            tag="nfl-analyzer-dev-size-test"
        )
        self.test_images.append(dev_image.id)
        
        prod_image, _ = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="production",
            tag="nfl-analyzer-prod-size-test"
        )
        self.test_images.append(prod_image.id)
        
        # Get image sizes
        dev_size = dev_image.attrs['Size']
        prod_size = prod_image.attrs['Size']
        
        # Production image should be reasonably sized
        # Convert to MB for easier assertion
        prod_size_mb = prod_size / (1024 * 1024)
        
        # Should be less than 1GB for a Python web app
        assert prod_size_mb < 1024, f"Production image too large: {prod_size_mb:.2f}MB"
        
        # Log sizes for monitoring
        print(f"Development image size: {dev_size / (1024 * 1024):.2f}MB")
        print(f"Production image size: {prod_size_mb:.2f}MB")


class TestDockerBuildSecurity:
    """Test Docker build security practices."""
    
    def setup_method(self):
        """Set up test environment."""
        self.docker_client = docker.from_env()
        self.test_images = []
    
    def teardown_method(self):
        """Clean up test resources."""
        for image_id in self.test_images:
            try:
                self.docker_client.images.remove(image_id, force=True)
            except docker.errors.ImageNotFound:
                pass
    
    def test_base_image_security(self):
        """Test base image security."""
        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            pytest.skip("Dockerfile not found")
        
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Should use specific version tags, not latest
        assert "python:3.11-slim" in dockerfile_content
        assert "FROM python:latest" not in dockerfile_content
        
        # Should use slim/alpine variants when possible
        assert "slim" in dockerfile_content or "alpine" in dockerfile_content
    
    def test_user_security(self):
        """Test user security in Docker images."""
        # Build production image
        image, _ = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="production",
            tag="nfl-analyzer-user-test"
        )
        self.test_images.append(image.id)
        
        # Test that container runs as non-root user
        container = self.docker_client.containers.run(
            image.id,
            command="id",
            remove=True,
            detach=False
        )
        
        output = container.decode().strip()
        # Should not be root (uid=0)
        assert "uid=0" not in output, "Container should not run as root"
        assert "app" in output, "Container should run as app user"
    
    def test_secrets_not_in_image(self):
        """Test that secrets are not baked into the image."""
        # Build production image
        image, _ = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="production",
            tag="nfl-analyzer-secrets-test"
        )
        self.test_images.append(image.id)
        
        # Check environment variables in image
        config = image.attrs['Config']
        env_vars = config.get('Env', [])
        
        # Should not contain hardcoded secrets
        for env_var in env_vars:
            env_var_lower = env_var.lower()
            # Check for common secret patterns
            if any(secret in env_var_lower for secret in ['password', 'secret', 'key', 'token']):
                # Should not have actual values, only variable names
                assert '=' not in env_var or env_var.split('=')[1] == '', \
                    f"Secret value found in image: {env_var}"
    
    def test_file_permissions(self):
        """Test file permissions in Docker image."""
        # Build production image
        image, _ = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="production",
            tag="nfl-analyzer-permissions-test"
        )
        self.test_images.append(image.id)
        
        # Check file permissions
        container = self.docker_client.containers.run(
            image.id,
            command="ls -la /app",
            remove=True,
            detach=False
        )
        
        output = container.decode()
        lines = output.strip().split('\n')
        
        # Check that app directory is owned by app user
        for line in lines:
            if '/app' in line or 'app.py' in line:
                # Should be owned by app user
                assert 'app app' in line, f"File not owned by app user: {line}"


class TestDockerComposeValidation:
    """Test Docker Compose configuration validation."""
    
    def test_compose_file_syntax(self):
        """Test Docker Compose files have valid syntax."""
        compose_files = [
            'docker-compose.yml',
            'docker-compose.dev.yml',
            'docker-compose.prod.yml'
        ]
        
        for compose_file in compose_files:
            if not Path(compose_file).exists():
                continue
            
            # Use docker-compose to validate syntax
            try:
                result = subprocess.run(
                    ['docker-compose', '-f', compose_file, 'config'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    pytest.fail(f"Invalid Docker Compose syntax in {compose_file}: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                pytest.fail(f"Docker Compose validation timed out for {compose_file}")
            except FileNotFoundError:
                pytest.skip("docker-compose command not available")
    
    def test_service_build_contexts(self):
        """Test service build contexts are valid."""
        compose_file = 'docker-compose.yml'
        if not Path(compose_file).exists():
            pytest.skip(f"{compose_file} not found")
        
        import yaml
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config.get('services', {})
        
        for service_name, service_config in services.items():
            build_config = service_config.get('build')
            if build_config:
                if isinstance(build_config, dict):
                    context = build_config.get('context', '.')
                    dockerfile = build_config.get('dockerfile', 'Dockerfile')
                    
                    # Check that build context exists
                    context_path = Path(context)
                    assert context_path.exists(), f"Build context not found for {service_name}: {context}"
                    
                    # Check that Dockerfile exists in context
                    dockerfile_path = context_path / dockerfile
                    assert dockerfile_path.exists(), f"Dockerfile not found for {service_name}: {dockerfile_path}"
    
    def test_environment_variable_references(self):
        """Test environment variable references are valid."""
        compose_files = ['docker-compose.yml', 'docker-compose.prod.yml']
        
        for compose_file in compose_files:
            if not Path(compose_file).exists():
                continue
            
            import yaml
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            services = compose_config.get('services', {})
            
            for service_name, service_config in services.items():
                environment = service_config.get('environment', [])
                
                if isinstance(environment, list):
                    for env_var in environment:
                        if isinstance(env_var, str) and '${' in env_var:
                            # Extract variable name
                            var_name = env_var.split('${')[1].split('}')[0].split(':-')[0]
                            
                            # Should be a valid environment variable name
                            assert var_name.replace('_', '').isalnum(), \
                                f"Invalid environment variable name in {service_name}: {var_name}"


class TestBuildPerformance:
    """Test Docker build performance and caching."""
    
    def setup_method(self):
        """Set up test environment."""
        self.docker_client = docker.from_env()
        self.test_images = []
    
    def teardown_method(self):
        """Clean up test resources."""
        for image_id in self.test_images:
            try:
                self.docker_client.images.remove(image_id, force=True)
            except docker.errors.ImageNotFound:
                pass
    
    def test_build_caching(self):
        """Test Docker build caching is effective."""
        import time
        
        # First build
        start_time = time.time()
        image1, _ = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="production",
            tag="nfl-analyzer-cache-test-1"
        )
        first_build_time = time.time() - start_time
        self.test_images.append(image1.id)
        
        # Second build (should use cache)
        start_time = time.time()
        image2, _ = self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            target="production",
            tag="nfl-analyzer-cache-test-2"
        )
        second_build_time = time.time() - start_time
        self.test_images.append(image2.id)
        
        # Second build should be significantly faster due to caching
        # Allow some variance but should be at least 50% faster
        assert second_build_time < first_build_time * 0.8, \
            f"Build caching not effective: {first_build_time:.2f}s vs {second_build_time:.2f}s"
        
        print(f"First build: {first_build_time:.2f}s, Second build: {second_build_time:.2f}s")
    
    def test_requirements_caching(self):
        """Test that requirements installation is cached properly."""
        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            pytest.skip("Dockerfile not found")
        
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Check that requirements.txt is copied before application code
        # This allows Docker to cache the pip install layer
        lines = dockerfile_content.split('\n')
        
        copy_requirements_line = -1
        copy_app_line = -1
        pip_install_line = -1
        
        for i, line in enumerate(lines):
            if 'COPY requirements.txt' in line:
                copy_requirements_line = i
            elif 'pip install' in line and 'requirements.txt' in line:
                pip_install_line = i
            elif 'COPY app/' in line or 'COPY . .' in line:
                copy_app_line = i
        
        # Requirements should be copied and installed before app code
        if copy_requirements_line >= 0 and copy_app_line >= 0 and pip_install_line >= 0:
            assert copy_requirements_line < pip_install_line < copy_app_line, \
                "Requirements should be installed before copying application code for better caching"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])