"""
Docker Environment Validation Script

Quick validation script to check if the Docker environment is properly
set up for running deployment tests.

Requirements covered: 8.1, 8.2
"""

import subprocess
import sys
import json
import docker
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DockerEnvironmentValidator:
    """Validates Docker environment for deployment tests."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.docker_client = None
    
    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("🔍 Validating Docker Environment for Deployment Tests")
        print("=" * 60)
        
        checks = [
            ("Docker Installation", self._check_docker_installation),
            ("Docker Compose", self._check_docker_compose),
            ("Docker Daemon", self._check_docker_daemon),
            ("Docker Permissions", self._check_docker_permissions),
            ("Required Files", self._check_required_files),
            ("Docker Images", self._check_existing_images),
            ("Docker Networks", self._check_docker_networks),
            ("Docker Volumes", self._check_docker_volumes),
            ("System Resources", self._check_system_resources),
        ]
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}:")
            try:
                check_func()
                print(f"   ✅ {check_name} - OK")
            except Exception as e:
                self.issues.append(f"{check_name}: {str(e)}")
                print(f"   ❌ {check_name} - FAILED: {e}")
        
        self._print_summary()
        return len(self.issues) == 0
    
    def _check_docker_installation(self):
        """Check if Docker is installed and accessible."""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise Exception("Docker command failed")
            
            version_info = result.stdout.strip()
            print(f"   📦 {version_info}")
            
            # Check Docker version (should be reasonably recent)
            if "Docker version" in version_info:
                version_part = version_info.split("Docker version ")[1].split(",")[0]
                major_version = int(version_part.split(".")[0])
                if major_version < 20:
                    self.warnings.append(f"Docker version {version_part} is quite old")
            
        except subprocess.TimeoutExpired:
            raise Exception("Docker command timed out")
        except FileNotFoundError:
            raise Exception("Docker not found - please install Docker")
        except Exception as e:
            raise Exception(f"Docker check failed: {e}")
    
    def _check_docker_compose(self):
        """Check if Docker Compose is available."""
        # Try docker-compose first
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_info = result.stdout.strip()
                print(f"   🔧 {version_info}")
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Try docker compose (newer syntax)
        try:
            result = subprocess.run(['docker', 'compose', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_info = result.stdout.strip()
                print(f"   🔧 Docker Compose (plugin): {version_info}")
                return
        except subprocess.TimeoutExpired:
            pass
        
        raise Exception("Docker Compose not found - please install Docker Compose")
    
    def _check_docker_daemon(self):
        """Check if Docker daemon is running."""
        try:
            self.docker_client = docker.from_env()
            info = self.docker_client.info()
            
            print(f"   🐳 Docker daemon running")
            print(f"   📊 Containers: {info.get('Containers', 0)}")
            print(f"   🖼️  Images: {info.get('Images', 0)}")
            
        except docker.errors.DockerException as e:
            raise Exception(f"Docker daemon not accessible: {e}")
    
    def _check_docker_permissions(self):
        """Check Docker permissions."""
        try:
            # Try to run a simple Docker command
            result = subprocess.run(['docker', 'ps'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                if "permission denied" in result.stderr.lower():
                    raise Exception("Permission denied - user may need to be in docker group")
                else:
                    raise Exception(f"Docker ps failed: {result.stderr}")
            
            print(f"   🔐 Docker permissions OK")
            
        except subprocess.TimeoutExpired:
            raise Exception("Docker ps command timed out")
        except FileNotFoundError:
            raise Exception("Docker command not found")
    
    def _check_required_files(self):
        """Check if required files exist."""
        required_files = [
            ('Dockerfile', 'Main application Dockerfile'),
            ('docker-compose.yml', 'Main Docker Compose file'),
            ('requirements.txt', 'Python requirements'),
            ('.dockerignore', 'Docker ignore file (recommended)'),
        ]
        
        optional_files = [
            ('docker-compose.dev.yml', 'Development Docker Compose'),
            ('docker-compose.prod.yml', 'Production Docker Compose'),
            ('frontend/Dockerfile', 'Frontend Dockerfile'),
            ('.env.example', 'Environment variables example'),
        ]
        
        for file_path, description in required_files:
            path = Path(file_path)
            if not path.exists():
                if file_path == '.dockerignore':
                    self.warnings.append(f"Missing {description}: {file_path}")
                    print(f"   ⚠️  Missing {description}: {file_path}")
                else:
                    raise Exception(f"Missing required file: {file_path}")
            else:
                print(f"   ✅ Found {description}")
        
        for file_path, description in optional_files:
            path = Path(file_path)
            if path.exists():
                print(f"   ✅ Found {description}")
            else:
                print(f"   ℹ️  Optional file not found: {description}")
    
    def _check_existing_images(self):
        """Check existing Docker images."""
        if not self.docker_client:
            return
        
        try:
            images = self.docker_client.images.list()
            
            # Look for application images
            app_images = [img for img in images 
                         if any(tag for tag in img.tags 
                               if 'nfl-analyzer' in tag or 'nfl_analyzer' in tag)]
            
            if app_images:
                print(f"   🖼️  Found {len(app_images)} application images")
                for img in app_images[:3]:  # Show first 3
                    tags = img.tags[0] if img.tags else img.id[:12]
                    size_mb = img.attrs['Size'] / (1024 * 1024)
                    print(f"      - {tags} ({size_mb:.1f}MB)")
            else:
                print(f"   ℹ️  No application images found (will be built during tests)")
            
            # Check for base images
            base_images = [img for img in images 
                          if any(tag for tag in img.tags 
                                if 'python:3.11' in tag or 'node:' in tag or 'nginx:' in tag)]
            
            if base_images:
                print(f"   📦 Found {len(base_images)} base images")
            
        except Exception as e:
            self.warnings.append(f"Could not check existing images: {e}")
    
    def _check_docker_networks(self):
        """Check Docker networks."""
        if not self.docker_client:
            return
        
        try:
            networks = self.docker_client.networks.list()
            
            # Check for application networks
            app_networks = [net for net in networks 
                           if 'nfl' in net.name.lower() or 'analyzer' in net.name.lower()]
            
            if app_networks:
                print(f"   🌐 Found {len(app_networks)} application networks")
                for net in app_networks:
                    print(f"      - {net.name}")
            else:
                print(f"   ℹ️  No application networks found (will be created as needed)")
            
            # Check default networks
            default_networks = ['bridge', 'host', 'none']
            existing_defaults = [net.name for net in networks if net.name in default_networks]
            print(f"   🔗 Default networks: {', '.join(existing_defaults)}")
            
        except Exception as e:
            self.warnings.append(f"Could not check Docker networks: {e}")
    
    def _check_docker_volumes(self):
        """Check Docker volumes."""
        if not self.docker_client:
            return
        
        try:
            volumes = self.docker_client.volumes.list()
            
            # Check for application volumes
            app_volumes = [vol for vol in volumes 
                          if 'nfl' in vol.name.lower() or 'mongodb' in vol.name.lower() 
                          or 'redis' in vol.name.lower()]
            
            if app_volumes:
                print(f"   💾 Found {len(app_volumes)} application volumes")
                for vol in app_volumes[:5]:  # Show first 5
                    print(f"      - {vol.name}")
            else:
                print(f"   ℹ️  No application volumes found (will be created as needed)")
            
        except Exception as e:
            self.warnings.append(f"Could not check Docker volumes: {e}")
    
    def _check_system_resources(self):
        """Check system resources."""
        try:
            import psutil
            
            # Check available memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            print(f"   💾 Memory: {memory_available_gb:.1f}GB available / {memory_gb:.1f}GB total")
            
            if memory_available_gb < 2:
                self.warnings.append("Low available memory (< 2GB) - may affect performance")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            print(f"   💿 Disk: {disk_free_gb:.1f}GB free / {disk_total_gb:.1f}GB total")
            
            if disk_free_gb < 5:
                self.warnings.append("Low disk space (< 5GB) - may affect Docker operations")
            
            # Check CPU
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            print(f"   🖥️  CPU: {cpu_count} cores, {cpu_percent:.1f}% current usage")
            
        except ImportError:
            print(f"   ℹ️  psutil not available - skipping detailed resource check")
        except Exception as e:
            self.warnings.append(f"Could not check system resources: {e}")
    
    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("🏁 VALIDATION SUMMARY")
        print("=" * 60)
        
        if not self.issues and not self.warnings:
            print("✅ All checks passed! Environment is ready for deployment tests.")
        elif not self.issues:
            print("✅ Environment is ready for deployment tests.")
            if self.warnings:
                print(f"⚠️  {len(self.warnings)} warnings (non-blocking):")
                for warning in self.warnings:
                    print(f"   - {warning}")
        else:
            print(f"❌ {len(self.issues)} issues found that need to be resolved:")
            for issue in self.issues:
                print(f"   - {issue}")
            
            if self.warnings:
                print(f"\n⚠️  {len(self.warnings)} additional warnings:")
                for warning in self.warnings:
                    print(f"   - {warning}")
        
        print("\n📚 Next Steps:")
        if not self.issues:
            print("   1. Run deployment tests: python tests/run_deployment_tests.py")
            print("   2. Or run specific tests: pytest tests/test_docker_builds.py -v")
        else:
            print("   1. Fix the issues listed above")
            print("   2. Re-run this validation: python tests/validate_docker_environment.py")
            print("   3. Then run deployment tests")


def main():
    """Main entry point."""
    validator = DockerEnvironmentValidator()
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()