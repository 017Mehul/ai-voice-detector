#!/usr/bin/env python3
"""
Production Configuration Validation Script

This script validates production environment configuration for the AI Voice Detection System.
It checks for security issues, performance settings, and deployment readiness.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re


class ProductionConfigValidator:
    """Validates production configuration for security and best practices"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        
        # Security validation rules
        self.security_rules = {
            "DEBUG": {
                "required": True,
                "expected": "false",
                "severity": "critical",
                "message": "DEBUG must be false in production"
            },
            "ENVIRONMENT": {
                "required": True,
                "expected": "production",
                "severity": "critical",
                "message": "ENVIRONMENT must be set to 'production'"
            },
            "LOG_LEVEL": {
                "required": True,
                "allowed": ["INFO", "WARNING", "ERROR", "CRITICAL"],
                "severity": "warning",
                "message": "LOG_LEVEL should be INFO or higher in production"
            },
            "CORS_ORIGINS": {
                "required": False,
                "pattern": r"^(?!\*$)",  # Should not be just "*"
                "severity": "warning",
                "message": "CORS_ORIGINS should specify exact domains, not '*'"
            },
            "API_KEY": {
                "required": False,
                "min_length": 16,
                "severity": "warning",
                "message": "API_KEY should be at least 16 characters long"
            },
            "JWT_SECRET": {
                "required": False,
                "min_length": 32,
                "severity": "warning",
                "message": "JWT_SECRET should be at least 32 characters long"
            }
        }
        
        # Performance validation rules
        self.performance_rules = {
            "API_WORKERS": {
                "type": "int",
                "min": 1,
                "max": 32,
                "recommended": (2, 8),
                "message": "API_WORKERS should be between 2-8 for optimal performance"
            },
            "MAX_AUDIO_SIZE_MB": {
                "type": "int",
                "min": 1,
                "max": 100,
                "recommended": (5, 15),
                "message": "MAX_AUDIO_SIZE_MB should be between 5-15 MB"
            },
            "RATE_LIMIT_REQUESTS": {
                "type": "int",
                "min": 1,
                "max": 10000,
                "recommended": (50, 500),
                "message": "RATE_LIMIT_REQUESTS should be between 50-500"
            },
            "REQUEST_TIMEOUT": {
                "type": "int",
                "min": 5,
                "max": 300,
                "recommended": (15, 60),
                "message": "REQUEST_TIMEOUT should be between 15-60 seconds"
            }
        }
        
        # Required files and paths
        self.required_paths = [
            "MODEL_PATH",
        ]
    
    def load_env_file(self, file_path: Path) -> Dict[str, str]:
        """Load environment variables from file"""
        env_vars = {}
        if not file_path.exists():
            return env_vars
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        try:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
                        except ValueError:
                            print(f"Warning: Invalid line {line_num} in {file_path}: {line}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return env_vars
    
    def validate_security(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate security-related configuration"""
        issues = []
        
        for var, rule in self.security_rules.items():
            value = config.get(var, "").strip()
            
            # Check if required variable is present
            if rule.get("required", False) and not value:
                issues.append({
                    "type": "security",
                    "severity": rule["severity"],
                    "variable": var,
                    "message": f"Required security variable {var} is missing",
                    "recommendation": f"Set {var} according to security requirements"
                })
                continue
            
            if not value:
                continue
            
            # Check expected value
            if "expected" in rule and value.lower() != rule["expected"].lower():
                issues.append({
                    "type": "security",
                    "severity": rule["severity"],
                    "variable": var,
                    "message": rule["message"],
                    "current_value": value,
                    "expected_value": rule["expected"],
                    "recommendation": f"Set {var}={rule['expected']}"
                })
            
            # Check allowed values
            if "allowed" in rule and value.upper() not in rule["allowed"]:
                issues.append({
                    "type": "security",
                    "severity": rule["severity"],
                    "variable": var,
                    "message": rule["message"],
                    "current_value": value,
                    "allowed_values": rule["allowed"],
                    "recommendation": f"Set {var} to one of: {', '.join(rule['allowed'])}"
                })
            
            # Check pattern matching
            if "pattern" in rule and not re.match(rule["pattern"], value):
                issues.append({
                    "type": "security",
                    "severity": rule["severity"],
                    "variable": var,
                    "message": rule["message"],
                    "current_value": value,
                    "recommendation": "Update value to meet security requirements"
                })
            
            # Check minimum length
            if "min_length" in rule and len(value) < rule["min_length"]:
                issues.append({
                    "type": "security",
                    "severity": rule["severity"],
                    "variable": var,
                    "message": rule["message"],
                    "current_length": len(value),
                    "min_length": rule["min_length"],
                    "recommendation": f"Use a {var} with at least {rule['min_length']} characters"
                })
        
        return issues
    
    def validate_performance(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate performance-related configuration"""
        issues = []
        
        for var, rule in self.performance_rules.items():
            value_str = config.get(var, "").strip()
            if not value_str:
                continue
            
            try:
                if rule["type"] == "int":
                    value = int(value_str)
                elif rule["type"] == "float":
                    value = float(value_str)
                else:
                    continue
                
                # Check bounds
                if "min" in rule and value < rule["min"]:
                    issues.append({
                        "type": "performance",
                        "severity": "warning",
                        "variable": var,
                        "message": f"{var} is below minimum recommended value",
                        "current_value": value,
                        "min_value": rule["min"],
                        "recommendation": f"Set {var} to at least {rule['min']}"
                    })
                
                if "max" in rule and value > rule["max"]:
                    issues.append({
                        "type": "performance",
                        "severity": "warning",
                        "variable": var,
                        "message": f"{var} exceeds maximum recommended value",
                        "current_value": value,
                        "max_value": rule["max"],
                        "recommendation": f"Set {var} to at most {rule['max']}"
                    })
                
                # Check recommended range
                if "recommended" in rule:
                    min_rec, max_rec = rule["recommended"]
                    if not (min_rec <= value <= max_rec):
                        issues.append({
                            "type": "performance",
                            "severity": "info",
                            "variable": var,
                            "message": rule["message"],
                            "current_value": value,
                            "recommended_range": f"{min_rec}-{max_rec}",
                            "recommendation": f"Consider setting {var} between {min_rec} and {max_rec}"
                        })
            
            except ValueError:
                issues.append({
                    "type": "performance",
                    "severity": "error",
                    "variable": var,
                    "message": f"{var} must be a valid {rule['type']}",
                    "current_value": value_str,
                    "recommendation": f"Set {var} to a valid {rule['type']} value"
                })
        
        return issues
    
    def validate_paths(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate file paths and directories"""
        issues = []
        
        for var in self.required_paths:
            path_str = config.get(var, "").strip()
            if not path_str:
                continue
            
            path = Path(path_str)
            
            # Check if path is absolute or relative to project root
            if not path.is_absolute():
                path = self.project_root / path
            
            if not path.exists():
                issues.append({
                    "type": "path",
                    "severity": "error",
                    "variable": var,
                    "message": f"Required file/directory does not exist: {path_str}",
                    "full_path": str(path),
                    "recommendation": f"Ensure {path_str} exists or update {var} to correct path"
                })
            elif var == "MODEL_PATH" and not path.is_file():
                issues.append({
                    "type": "path",
                    "severity": "error",
                    "variable": var,
                    "message": f"Model path should point to a file, not directory: {path_str}",
                    "recommendation": f"Update {var} to point to model file (e.g., model.joblib)"
                })
        
        return issues
    
    def validate_consistency(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate configuration consistency and logical relationships"""
        issues = []
        
        # Check audio duration consistency
        try:
            min_duration = int(config.get("MIN_AUDIO_DURATION_SECONDS", "2"))
            max_duration = int(config.get("MAX_AUDIO_DURATION_SECONDS", "30"))
            
            if min_duration >= max_duration:
                issues.append({
                    "type": "consistency",
                    "severity": "error",
                    "message": "MIN_AUDIO_DURATION_SECONDS must be less than MAX_AUDIO_DURATION_SECONDS",
                    "min_duration": min_duration,
                    "max_duration": max_duration,
                    "recommendation": "Adjust audio duration limits to ensure min < max"
                })
        except ValueError:
            pass  # Will be caught by performance validation
        
        # Check rate limiting consistency
        try:
            rate_enabled = config.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
            rate_requests = int(config.get("RATE_LIMIT_REQUESTS", "100"))
            rate_window = int(config.get("RATE_LIMIT_WINDOW", "60"))
            
            if rate_enabled and (rate_requests <= 0 or rate_window <= 0):
                issues.append({
                    "type": "consistency",
                    "severity": "error",
                    "message": "Rate limiting is enabled but requests or window is invalid",
                    "rate_requests": rate_requests,
                    "rate_window": rate_window,
                    "recommendation": "Set positive values for RATE_LIMIT_REQUESTS and RATE_LIMIT_WINDOW"
                })
        except ValueError:
            pass
        
        # Check CORS configuration
        cors_enabled = config.get("ENABLE_CORS", "true").lower() == "true"
        cors_origins = config.get("CORS_ORIGINS", "*")
        
        if cors_enabled and not cors_origins:
            issues.append({
                "type": "consistency",
                "severity": "warning",
                "message": "CORS is enabled but no origins are specified",
                "recommendation": "Set CORS_ORIGINS to allowed domains or disable CORS"
            })
        
        return issues
    
    def generate_report(self, issues: List[Dict[str, Any]]) -> str:
        """Generate a formatted validation report"""
        if not issues:
            return "✅ All validation checks passed! Configuration is production-ready."
        
        # Group issues by severity
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        error_issues = [i for i in issues if i.get("severity") == "error"]
        warning_issues = [i for i in issues if i.get("severity") == "warning"]
        info_issues = [i for i in issues if i.get("severity") == "info"]
        
        report = []
        report.append("🔍 Production Configuration Validation Report")
        report.append("=" * 50)
        
        if critical_issues:
            report.append("\n🚨 CRITICAL ISSUES (Must Fix Before Deployment):")
            for issue in critical_issues:
                report.append(f"  ❌ {issue['message']}")
                if 'recommendation' in issue:
                    report.append(f"     💡 {issue['recommendation']}")
        
        if error_issues:
            report.append("\n❌ ERRORS (Should Fix Before Deployment):")
            for issue in error_issues:
                report.append(f"  • {issue['message']}")
                if 'recommendation' in issue:
                    report.append(f"    💡 {issue['recommendation']}")
        
        if warning_issues:
            report.append("\n⚠️  WARNINGS (Recommended to Fix):")
            for issue in warning_issues:
                report.append(f"  • {issue['message']}")
                if 'recommendation' in issue:
                    report.append(f"    💡 {issue['recommendation']}")
        
        if info_issues:
            report.append("\n💡 SUGGESTIONS (Optional Improvements):")
            for issue in info_issues:
                report.append(f"  • {issue['message']}")
                if 'recommendation' in issue:
                    report.append(f"    💡 {issue['recommendation']}")
        
        # Summary
        report.append(f"\n📊 SUMMARY:")
        report.append(f"  Critical Issues: {len(critical_issues)}")
        report.append(f"  Errors: {len(error_issues)}")
        report.append(f"  Warnings: {len(warning_issues)}")
        report.append(f"  Suggestions: {len(info_issues)}")
        
        if critical_issues or error_issues:
            report.append(f"\n❌ Configuration is NOT ready for production deployment.")
            report.append(f"   Please fix critical issues and errors before deploying.")
        elif warning_issues:
            report.append(f"\n⚠️  Configuration has warnings but can be deployed.")
            report.append(f"   Consider fixing warnings for better security and performance.")
        else:
            report.append(f"\n✅ Configuration is production-ready!")
        
        return "\n".join(report)
    
    def validate_config_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate a configuration file and return success status and report"""
        if not file_path.exists():
            return False, f"❌ Configuration file not found: {file_path}"
        
        config = self.load_env_file(file_path)
        
        if not config:
            return False, f"❌ No configuration variables found in {file_path}"
        
        # Run all validations
        all_issues = []
        all_issues.extend(self.validate_security(config))
        all_issues.extend(self.validate_performance(config))
        all_issues.extend(self.validate_paths(config))
        all_issues.extend(self.validate_consistency(config))
        
        report = self.generate_report(all_issues)
        
        # Determine if configuration is acceptable for production
        critical_issues = [i for i in all_issues if i.get("severity") == "critical"]
        error_issues = [i for i in all_issues if i.get("severity") == "error"]
        
        is_production_ready = len(critical_issues) == 0 and len(error_issues) == 0
        
        return is_production_ready, report


def main():
    """Main entry point for configuration validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate production configuration")
    parser.add_argument("--config", type=str, default=".env",
                       help="Path to configuration file (default: .env)")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")
    parser.add_argument("--strict", action="store_true",
                       help="Treat warnings as errors")
    
    args = parser.parse_args()
    
    validator = ProductionConfigValidator()
    config_path = Path(args.config)
    
    is_ready, report = validator.validate_config_file(config_path)
    
    if args.json:
        # Output JSON format for programmatic use
        result = {
            "file": str(config_path),
            "production_ready": is_ready,
            "report": report
        }
        print(json.dumps(result, indent=2))
    else:
        # Output human-readable format
        print(report)
    
    # Exit with appropriate code
    if is_ready:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()