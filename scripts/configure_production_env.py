#!/usr/bin/env python3
"""
Production Environment Configuration Script

This script helps configure environment variables for production deployment
of the AI Voice Detection System. It validates settings, generates secure
defaults, and creates deployment-ready configuration files.
"""

import os
import sys
import argparse
import secrets
import string
from pathlib import Path
from typing import Dict, Any, Optional, List
import json


class ProductionEnvConfigurator:
    """Configures production environment variables with validation and security checks"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.env_production_path = self.project_root / ".env.production"
        self.env_local_path = self.project_root / ".env"
        
        # Required environment variables for production
        self.required_vars = {
            "ENVIRONMENT": "production",
            "DEBUG": "false",
            "LOG_LEVEL": "INFO",
            "API_HOST": "0.0.0.0",
            "API_PORT": "8000",
            "MODEL_PATH": "./models/production/model.joblib",
        }
        
        # Security-sensitive variables that should be generated or provided
        self.security_vars = [
            "API_KEY",
            "JWT_SECRET",
            "DATABASE_URL",
            "SENTRY_DSN",
            "REDIS_URL",
        ]
        
        # Performance tuning recommendations based on deployment size
        self.deployment_profiles = {
            "small": {
                "API_WORKERS": "2",
                "MAX_CONCURRENT_REQUESTS": "50",
                "RATE_LIMIT_REQUESTS": "50",
                "MAX_AUDIO_SIZE_MB": "5",
            },
            "medium": {
                "API_WORKERS": "4",
                "MAX_CONCURRENT_REQUESTS": "100",
                "RATE_LIMIT_REQUESTS": "100",
                "MAX_AUDIO_SIZE_MB": "10",
            },
            "large": {
                "API_WORKERS": "8",
                "MAX_CONCURRENT_REQUESTS": "200",
                "RATE_LIMIT_REQUESTS": "200",
                "MAX_AUDIO_SIZE_MB": "15",
            }
        }
    
    def generate_secure_key(self, length: int = 32) -> str:
        """Generate a cryptographically secure random key"""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def validate_environment_vars(self, env_vars: Dict[str, str]) -> List[str]:
        """Validate environment variables and return list of issues"""
        issues = []
        
        # Check required variables
        for var, expected_value in self.required_vars.items():
            if var not in env_vars:
                issues.append(f"Missing required variable: {var}")
            elif var == "ENVIRONMENT" and env_vars[var] != "production":
                issues.append(f"ENVIRONMENT must be 'production', got '{env_vars[var]}'")
            elif var == "DEBUG" and env_vars[var].lower() != "false":
                issues.append(f"DEBUG must be 'false' in production, got '{env_vars[var]}'")
        
        # Validate numeric values
        numeric_vars = {
            "API_PORT": (1, 65535),
            "API_WORKERS": (1, 32),
            "MAX_AUDIO_SIZE_MB": (1, 100),
            "MAX_AUDIO_DURATION_SECONDS": (1, 300),
            "MIN_AUDIO_DURATION_SECONDS": (1, 10),
            "RATE_LIMIT_REQUESTS": (1, 10000),
            "RATE_LIMIT_WINDOW": (1, 3600),
        }
        
        for var, (min_val, max_val) in numeric_vars.items():
            if var in env_vars:
                try:
                    value = int(env_vars[var])
                    if not (min_val <= value <= max_val):
                        issues.append(f"{var} must be between {min_val} and {max_val}, got {value}")
                except ValueError:
                    issues.append(f"{var} must be a valid integer, got '{env_vars[var]}'")
        
        # Validate float values
        float_vars = {
            "MODEL_CONFIDENCE_THRESHOLD": (0.0, 1.0),
        }
        
        for var, (min_val, max_val) in float_vars.items():
            if var in env_vars:
                try:
                    value = float(env_vars[var])
                    if not (min_val <= value <= max_val):
                        issues.append(f"{var} must be between {min_val} and {max_val}, got {value}")
                except ValueError:
                    issues.append(f"{var} must be a valid float, got '{env_vars[var]}'")
        
        # Check model path exists
        if "MODEL_PATH" in env_vars:
            model_path = Path(env_vars["MODEL_PATH"])
            if not model_path.exists() and not model_path.is_absolute():
                # Check relative to project root
                full_path = self.project_root / model_path
                if not full_path.exists():
                    issues.append(f"Model file not found: {env_vars['MODEL_PATH']}")
        
        # Security checks
        if env_vars.get("CORS_ORIGINS") == "*":
            issues.append("WARNING: CORS_ORIGINS set to '*' allows all origins. Specify exact domains in production.")
        
        if env_vars.get("LOG_LEVEL") == "DEBUG":
            issues.append("WARNING: LOG_LEVEL set to DEBUG may expose sensitive information in production.")
        
        return issues
    
    def load_env_file(self, file_path: Path) -> Dict[str, str]:
        """Load environment variables from .env file"""
        env_vars = {}
        if file_path.exists():
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        return env_vars
    
    def create_production_config(self, deployment_profile: str = "medium", 
                               custom_vars: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Create production configuration with specified profile"""
        # Start with base production template
        config = self.load_env_file(self.env_production_path)
        
        # Apply deployment profile
        if deployment_profile in self.deployment_profiles:
            config.update(self.deployment_profiles[deployment_profile])
        
        # Apply custom variables
        if custom_vars:
            config.update(custom_vars)
        
        # Generate secure keys if not provided
        if "API_KEY" not in config or not config["API_KEY"]:
            config["API_KEY"] = self.generate_secure_key(32)
        
        if "JWT_SECRET" not in config or not config["JWT_SECRET"]:
            config["JWT_SECRET"] = self.generate_secure_key(64)
        
        return config
    
    def write_env_file(self, config: Dict[str, str], output_path: Path):
        """Write configuration to .env file"""
        with open(output_path, 'w') as f:
            f.write("# AI Voice Detection System - Production Environment Configuration\n")
            f.write("# Generated by configure_production_env.py\n")
            f.write("# DO NOT COMMIT THIS FILE TO VERSION CONTROL\n\n")
            
            # Group variables by category
            categories = {
                "Core Application": ["ENVIRONMENT", "DEBUG", "LOG_LEVEL", "ENABLE_DETAILED_LOGGING"],
                "API Server": ["API_HOST", "API_PORT", "API_WORKERS", "REQUEST_TIMEOUT", "MAX_REQUEST_SIZE"],
                "CORS": ["ENABLE_CORS", "CORS_ORIGINS", "CORS_METHODS", "CORS_HEADERS"],
                "Model": ["MODEL_PATH", "MODEL_CONFIDENCE_THRESHOLD", "ENABLE_FEATURE_IMPORTANCE"],
                "Audio Processing": ["MAX_AUDIO_SIZE_MB", "MAX_AUDIO_DURATION_SECONDS", "MIN_AUDIO_DURATION_SECONDS"],
                "Rate Limiting": ["RATE_LIMIT_ENABLED", "RATE_LIMIT_REQUESTS", "RATE_LIMIT_WINDOW"],
                "Monitoring": ["HEALTH_CHECK_ENABLED", "METRICS_ENABLED", "PROMETHEUS_PORT"],
                "Security": ["API_KEY", "JWT_SECRET", "ENABLE_REQUEST_TRACKING"],
            }
            
            for category, vars_in_category in categories.items():
                f.write(f"# {category}\n")
                for var in vars_in_category:
                    if var in config:
                        f.write(f"{var}={config[var]}\n")
                f.write("\n")
            
            # Write remaining variables
            written_vars = set()
            for vars_in_category in categories.values():
                written_vars.update(vars_in_category)
            
            remaining_vars = set(config.keys()) - written_vars
            if remaining_vars:
                f.write("# Additional Configuration\n")
                for var in sorted(remaining_vars):
                    f.write(f"{var}={config[var]}\n")
    
    def create_docker_env_file(self, config: Dict[str, str], output_path: Path):
        """Create Docker-compatible environment file"""
        with open(output_path, 'w') as f:
            for key, value in sorted(config.items()):
                # Skip comments and empty values
                if value and not key.startswith('#'):
                    f.write(f"{key}={value}\n")
    
    def create_kubernetes_configmap(self, config: Dict[str, str], output_path: Path, 
                                  namespace: str = "default", name: str = "ai-voice-detector-config"):
        """Create Kubernetes ConfigMap YAML"""
        # Separate sensitive and non-sensitive config
        configmap_data = {}
        secret_data = {}
        
        for key, value in config.items():
            if key in self.security_vars or "SECRET" in key or "PASSWORD" in key or "KEY" in key:
                secret_data[key] = value
            else:
                configmap_data[key] = value
        
        yaml_content = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {name}
  namespace: {namespace}
data:
"""
        for key, value in sorted(configmap_data.items()):
            yaml_content += f'  {key}: "{value}"\n'
        
        if secret_data:
            yaml_content += f"""---
apiVersion: v1
kind: Secret
metadata:
  name: {name}-secrets
  namespace: {namespace}
type: Opaque
stringData:
"""
            for key, value in sorted(secret_data.items()):
                yaml_content += f'  {key}: "{value}"\n'
        
        with open(output_path, 'w') as f:
            f.write(yaml_content)
    
    def validate_production_readiness(self, config: Dict[str, str]) -> bool:
        """Validate that configuration is ready for production deployment"""
        issues = self.validate_environment_vars(config)
        
        if issues:
            print("❌ Production readiness validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("✅ Production configuration validation passed")
        return True
    
    def interactive_setup(self):
        """Interactive setup for production configuration"""
        print("🚀 AI Voice Detection System - Production Environment Setup")
        print("=" * 60)
        
        # Get deployment profile
        print("\nSelect deployment profile:")
        print("1. Small (2 workers, 50 req/min)")
        print("2. Medium (4 workers, 100 req/min)")
        print("3. Large (8 workers, 200 req/min)")
        
        while True:
            choice = input("Enter choice (1-3) [2]: ").strip() or "2"
            if choice in ["1", "2", "3"]:
                profiles = ["small", "medium", "large"]
                deployment_profile = profiles[int(choice) - 1]
                break
            print("Invalid choice. Please enter 1, 2, or 3.")
        
        # Get custom domain for CORS
        cors_origins = input("Enter allowed CORS origins (comma-separated) [*]: ").strip() or "*"
        
        # Get model path
        model_path = input("Enter model path [./models/production/model.joblib]: ").strip() or "./models/production/model.joblib"
        
        # Create configuration
        custom_vars = {
            "CORS_ORIGINS": cors_origins,
            "MODEL_PATH": model_path,
        }
        
        config = self.create_production_config(deployment_profile, custom_vars)
        
        # Validate configuration
        if not self.validate_production_readiness(config):
            print("\n❌ Configuration validation failed. Please fix the issues and try again.")
            return False
        
        # Write configuration files
        env_file = self.project_root / ".env"
        docker_env_file = self.project_root / ".env.docker"
        k8s_config_file = self.project_root / "k8s-config.yaml"
        
        self.write_env_file(config, env_file)
        self.create_docker_env_file(config, docker_env_file)
        self.create_kubernetes_configmap(config, k8s_config_file)
        
        print(f"\n✅ Production configuration created successfully!")
        print(f"📁 Files created:")
        print(f"  - {env_file} (Main environment file)")
        print(f"  - {docker_env_file} (Docker environment file)")
        print(f"  - {k8s_config_file} (Kubernetes ConfigMap)")
        
        print(f"\n🔐 Security Notes:")
        print(f"  - Generated API_KEY: {config.get('API_KEY', 'Not generated')[:8]}...")
        print(f"  - Generated JWT_SECRET: {config.get('JWT_SECRET', 'Not generated')[:8]}...")
        print(f"  - DO NOT commit .env files to version control")
        print(f"  - Store secrets securely in your deployment environment")
        
        return True


def main():
    """Main entry point for production environment configuration"""
    parser = argparse.ArgumentParser(description="Configure production environment for AI Voice Detection System")
    parser.add_argument("--profile", choices=["small", "medium", "large"], default="medium",
                       help="Deployment profile (default: medium)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive setup")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing .env file")
    parser.add_argument("--output", type=str, default=".env",
                       help="Output file path (default: .env)")
    
    args = parser.parse_args()
    
    configurator = ProductionEnvConfigurator()
    
    if args.validate:
        # Validate existing configuration
        env_file = Path(args.output)
        if not env_file.exists():
            print(f"❌ Environment file not found: {env_file}")
            sys.exit(1)
        
        config = configurator.load_env_file(env_file)
        if configurator.validate_production_readiness(config):
            print("✅ Environment configuration is production-ready")
            sys.exit(0)
        else:
            print("❌ Environment configuration has issues")
            sys.exit(1)
    
    elif args.interactive:
        # Run interactive setup
        if configurator.interactive_setup():
            sys.exit(0)
        else:
            sys.exit(1)
    
    else:
        # Create configuration with specified profile
        config = configurator.create_production_config(args.profile)
        
        if configurator.validate_production_readiness(config):
            output_path = Path(args.output)
            configurator.write_env_file(config, output_path)
            print(f"✅ Production configuration created: {output_path}")
        else:
            print("❌ Configuration validation failed")
            sys.exit(1)


if __name__ == "__main__":
    main()