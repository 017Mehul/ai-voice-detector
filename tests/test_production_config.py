"""
Tests for production environment configuration

This module tests the production configuration system including
validation, security checks, and deployment readiness.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.configure_production_env import ProductionEnvConfigurator
from scripts.validate_production_config import ProductionConfigValidator


class TestProductionEnvConfigurator:
    """Test production environment configurator"""
    
    def setup_method(self):
        """Set up test environment"""
        self.configurator = ProductionEnvConfigurator()
    
    def test_generate_secure_key(self):
        """Test secure key generation"""
        key1 = self.configurator.generate_secure_key(32)
        key2 = self.configurator.generate_secure_key(32)
        
        # Keys should be different
        assert key1 != key2
        
        # Keys should have correct length
        assert len(key1) == 32
        assert len(key2) == 32
        
        # Keys should contain valid characters
        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*")
        assert all(c in valid_chars for c in key1)
        assert all(c in valid_chars for c in key2)
    
    def test_validate_environment_vars_valid_config(self):
        """Test validation with valid configuration"""
        valid_config = {
            "ENVIRONMENT": "production",
            "DEBUG": "false",
            "LOG_LEVEL": "INFO",
            "API_HOST": "0.0.0.0",
            "API_PORT": "8000",
            "MODEL_PATH": "./models/production/model.joblib",
            "API_WORKERS": "4",
            "MAX_AUDIO_SIZE_MB": "10",
            "RATE_LIMIT_REQUESTS": "100",
            "RATE_LIMIT_WINDOW": "60",
            "MODEL_CONFIDENCE_THRESHOLD": "0.5",
        }
        
        issues = self.configurator.validate_environment_vars(valid_config)
        
        # Should have minimal issues (maybe model path warning)
        critical_issues = [i for i in issues if "Missing required variable" in i]
        assert len(critical_issues) == 0
    
    def test_validate_environment_vars_missing_required(self):
        """Test validation with missing required variables"""
        invalid_config = {
            "DEBUG": "true",  # Wrong value
            "LOG_LEVEL": "INFO",
        }
        
        issues = self.configurator.validate_environment_vars(invalid_config)
        
        # Should have issues for missing required variables
        missing_env = any("Missing required variable: ENVIRONMENT" in issue for issue in issues)
        missing_host = any("Missing required variable: API_HOST" in issue for issue in issues)
        debug_issue = any("DEBUG must be 'false' in production" in issue for issue in issues)
        
        assert missing_env
        assert missing_host
        assert debug_issue
    
    def test_validate_environment_vars_invalid_values(self):
        """Test validation with invalid values"""
        invalid_config = {
            "ENVIRONMENT": "production",
            "DEBUG": "false",
            "LOG_LEVEL": "INVALID",
            "API_HOST": "0.0.0.0",
            "API_PORT": "99999",  # Invalid port
            "MODEL_PATH": "./models/production/model.joblib",
            "API_WORKERS": "0",  # Invalid worker count
            "MODEL_CONFIDENCE_THRESHOLD": "2.0",  # Invalid threshold
        }
        
        issues = self.configurator.validate_environment_vars(invalid_config)
        
        # Should have issues for invalid values
        port_issue = any("API_PORT must be between 1 and 65535" in issue for issue in issues)
        worker_issue = any("API_WORKERS must be between 1 and 32" in issue for issue in issues)
        threshold_issue = any("MODEL_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0" in issue for issue in issues)
        
        assert port_issue
        assert worker_issue
        assert threshold_issue
    
    def test_create_production_config(self):
        """Test production configuration creation"""
        config = self.configurator.create_production_config("medium")
        
        # Should have required variables
        assert config["ENVIRONMENT"] == "production"
        assert config["DEBUG"] == "false"
        assert config["API_WORKERS"] == "4"  # Medium profile
        assert config["MAX_CONCURRENT_REQUESTS"] == "100"  # Medium profile
        
        # Should have generated secure keys
        assert "API_KEY" in config
        assert "JWT_SECRET" in config
        assert len(config["API_KEY"]) == 32
        assert len(config["JWT_SECRET"]) == 64
    
    def test_deployment_profiles(self):
        """Test different deployment profiles"""
        small_config = self.configurator.create_production_config("small")
        medium_config = self.configurator.create_production_config("medium")
        large_config = self.configurator.create_production_config("large")
        
        # Check worker scaling
        assert int(small_config["API_WORKERS"]) < int(medium_config["API_WORKERS"])
        assert int(medium_config["API_WORKERS"]) < int(large_config["API_WORKERS"])
        
        # Check request limits scaling
        assert int(small_config["RATE_LIMIT_REQUESTS"]) < int(medium_config["RATE_LIMIT_REQUESTS"])
        assert int(medium_config["RATE_LIMIT_REQUESTS"]) < int(large_config["RATE_LIMIT_REQUESTS"])


class TestProductionConfigValidator:
    """Test production configuration validator"""
    
    def setup_method(self):
        """Set up test environment"""
        self.validator = ProductionConfigValidator()
    
    def test_validate_security_valid_config(self):
        """Test security validation with valid configuration"""
        valid_config = {
            "DEBUG": "false",
            "ENVIRONMENT": "production",
            "LOG_LEVEL": "INFO",
            "CORS_ORIGINS": "https://example.com",
            "API_KEY": "a" * 20,  # Long enough
            "JWT_SECRET": "b" * 40,  # Long enough
        }
        
        issues = self.validator.validate_security(valid_config)
        
        # Should have no critical security issues
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        assert len(critical_issues) == 0
    
    def test_validate_security_invalid_config(self):
        """Test security validation with invalid configuration"""
        invalid_config = {
            "DEBUG": "true",  # Should be false
            "ENVIRONMENT": "development",  # Should be production
            "LOG_LEVEL": "DEBUG",  # Warning in production
            "CORS_ORIGINS": "*",  # Warning for wildcard
            "API_KEY": "short",  # Too short
        }
        
        issues = self.validator.validate_security(invalid_config)
        
        # Should have critical issues
        debug_issue = any(i.get("severity") == "critical" and "DEBUG" in i.get("variable", "") for i in issues)
        env_issue = any(i.get("severity") == "critical" and "ENVIRONMENT" in i.get("variable", "") for i in issues)
        
        assert debug_issue
        assert env_issue
    
    def test_validate_performance_valid_config(self):
        """Test performance validation with valid configuration"""
        valid_config = {
            "API_WORKERS": "4",
            "MAX_AUDIO_SIZE_MB": "10",
            "RATE_LIMIT_REQUESTS": "100",
            "REQUEST_TIMEOUT": "30",
        }
        
        issues = self.validator.validate_performance(valid_config)
        
        # Should have no errors, maybe some info suggestions
        error_issues = [i for i in issues if i.get("severity") == "error"]
        assert len(error_issues) == 0
    
    def test_validate_performance_invalid_config(self):
        """Test performance validation with invalid configuration"""
        invalid_config = {
            "API_WORKERS": "invalid",  # Not a number
            "MAX_AUDIO_SIZE_MB": "0",  # Too low
            "RATE_LIMIT_REQUESTS": "99999",  # Too high
            "REQUEST_TIMEOUT": "1",  # Too low
        }
        
        issues = self.validator.validate_performance(invalid_config)
        
        # Should have error issues
        worker_issue = any("API_WORKERS" in i.get("variable", "") and i.get("severity") == "error" for i in issues)
        size_issue = any("MAX_AUDIO_SIZE_MB" in i.get("variable", "") for i in issues)
        
        assert worker_issue
        assert size_issue
    
    def test_validate_consistency_valid_config(self):
        """Test consistency validation with valid configuration"""
        valid_config = {
            "MIN_AUDIO_DURATION_SECONDS": "2",
            "MAX_AUDIO_DURATION_SECONDS": "30",
            "RATE_LIMIT_ENABLED": "true",
            "RATE_LIMIT_REQUESTS": "100",
            "RATE_LIMIT_WINDOW": "60",
            "ENABLE_CORS": "true",
            "CORS_ORIGINS": "https://example.com",
        }
        
        issues = self.validator.validate_consistency(valid_config)
        
        # Should have no consistency issues
        error_issues = [i for i in issues if i.get("severity") == "error"]
        assert len(error_issues) == 0
    
    def test_validate_consistency_invalid_config(self):
        """Test consistency validation with invalid configuration"""
        invalid_config = {
            "MIN_AUDIO_DURATION_SECONDS": "30",  # Greater than max
            "MAX_AUDIO_DURATION_SECONDS": "10",  # Less than min
            "RATE_LIMIT_ENABLED": "true",
            "RATE_LIMIT_REQUESTS": "0",  # Invalid when enabled
            "RATE_LIMIT_WINDOW": "-1",  # Invalid when enabled
        }
        
        issues = self.validator.validate_consistency(invalid_config)
        
        # Should have consistency issues
        duration_issue = any("MIN_AUDIO_DURATION_SECONDS must be less than MAX_AUDIO_DURATION_SECONDS" in i.get("message", "") for i in issues)
        rate_limit_issue = any("Rate limiting is enabled but requests or window is invalid" in i.get("message", "") for i in issues)
        
        assert duration_issue
        assert rate_limit_issue
    
    def test_generate_report_no_issues(self):
        """Test report generation with no issues"""
        issues = []
        report = self.validator.generate_report(issues)
        
        assert "✅ All validation checks passed!" in report
        assert "Configuration is production-ready" in report
    
    def test_generate_report_with_issues(self):
        """Test report generation with various issues"""
        issues = [
            {
                "type": "security",
                "severity": "critical",
                "message": "DEBUG must be false in production",
                "recommendation": "Set DEBUG=false"
            },
            {
                "type": "performance",
                "severity": "warning",
                "message": "API_WORKERS should be optimized",
                "recommendation": "Consider adjusting worker count"
            },
            {
                "type": "consistency",
                "severity": "info",
                "message": "Configuration could be improved",
                "recommendation": "Review settings"
            }
        ]
        
        report = self.validator.generate_report(issues)
        
        assert "🚨 CRITICAL ISSUES" in report
        assert "⚠️  WARNINGS" in report
        assert "💡 SUGGESTIONS" in report
        assert "Configuration is NOT ready for production" in report
    
    def test_load_env_file_valid(self):
        """Test loading valid environment file"""
        env_content = """# Test environment file
DEBUG=false
ENVIRONMENT=production
API_PORT=8000
# Comment line
EMPTY_LINE=

VALID_VAR=value
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            f.flush()
            
            config = self.validator.load_env_file(Path(f.name))
            
            assert config["DEBUG"] == "false"
            assert config["ENVIRONMENT"] == "production"
            assert config["API_PORT"] == "8000"
            assert config["VALID_VAR"] == "value"
            # Empty values are loaded but can be filtered later if needed
            assert config.get("EMPTY_LINE") == ""
        
        # Clean up
        os.unlink(f.name)
    
    def test_validate_config_file_production_ready(self):
        """Test validation of production-ready configuration file"""
        production_config = """ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
MODEL_PATH=./models/production/model.joblib
CORS_ORIGINS=https://example.com
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(production_config)
            f.flush()
            
            # Mock the model path check since we don't have actual model file
            with patch.object(Path, 'exists', return_value=True):
                is_ready, report = self.validator.validate_config_file(Path(f.name))
            
            # Should be production ready (or close to it)
            assert "✅" in report or "Configuration has warnings but can be deployed" in report
        
        # Clean up
        os.unlink(f.name)
    
    def test_validate_config_file_not_ready(self):
        """Test validation of non-production-ready configuration file"""
        dev_config = """ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
API_HOST=localhost
API_PORT=8000
CORS_ORIGINS=*
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(dev_config)
            f.flush()
            
            is_ready, report = self.validator.validate_config_file(Path(f.name))
            
            # Should not be production ready
            assert not is_ready
            assert "🚨 CRITICAL ISSUES" in report or "❌ ERRORS" in report
        
        # Clean up
        os.unlink(f.name)


class TestConfigurationIntegration:
    """Integration tests for configuration system"""
    
    def test_configure_and_validate_workflow(self):
        """Test complete configure and validate workflow"""
        configurator = ProductionEnvConfigurator()
        validator = ProductionConfigValidator()
        
        # Create production configuration
        config = configurator.create_production_config("medium")
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            for key, value in config.items():
                f.write(f"{key}={value}\n")
            f.flush()
            
            # Mock model path existence
            with patch.object(Path, 'exists', return_value=True):
                is_ready, report = validator.validate_config_file(Path(f.name))
            
            # Configuration should be valid or have only minor warnings
            assert is_ready or "warnings but can be deployed" in report
        
        # Clean up
        os.unlink(f.name)
    
    def test_deployment_profile_validation(self):
        """Test that all deployment profiles create valid configurations"""
        configurator = ProductionEnvConfigurator()
        validator = ProductionConfigValidator()
        
        profiles = ["small", "medium", "large"]
        
        for profile in profiles:
            config = configurator.create_production_config(profile)
            
            # Validate the configuration
            issues = validator.validate_security(config)
            issues.extend(validator.validate_performance(config))
            issues.extend(validator.validate_consistency(config))
            
            # Should have no critical issues
            critical_issues = [i for i in issues if i.get("severity") == "critical"]
            error_issues = [i for i in issues if i.get("severity") == "error"]
            
            assert len(critical_issues) == 0, f"Profile {profile} has critical issues: {critical_issues}"
            assert len(error_issues) == 0, f"Profile {profile} has error issues: {error_issues}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])