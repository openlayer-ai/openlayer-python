#!/usr/bin/env python3
"""
CI Test: Integration modules conditional import handling.

This test ensures that all integration modules in src/openlayer/lib/integrations/
handle optional dependencies correctly:
1. Can be imported when dependency is not available
2. Provide helpful error messages when trying to use without dependency
3. Do not have type annotation errors
4. Follow consistent patterns for conditional imports

This prevents regressions in conditional import handling across all integrations.
"""

import sys
import tempfile
import textwrap
import subprocess
from typing import List, Tuple
from pathlib import Path

# Note: pytest is imported automatically when running via pytest
# This file can also be run standalone for manual testing


# Mapping of integration modules to their optional dependencies
INTEGRATION_DEPENDENCIES = {
    "openai_agents": ["agents"],
    "openai_tracer": ["openai"],
    "async_openai_tracer": ["openai"],
    "anthropic_tracer": ["anthropic"],
    "mistral_tracer": ["mistralai"],
    "groq_tracer": ["groq"],
    "langchain_callback": ["langchain", "langchain_core", "langchain_community"],
}

# Expected patterns for integration modules
EXPECTED_PATTERNS = {
    "availability_flag": True,  # Should have HAVE_<LIB> flag
    "helpful_error": True,      # Should give helpful error when instantiating without dependency
    "graceful_import": True,    # Should import without errors when dependency missing
}


def create_import_blocker_script(blocked_packages: List[str]) -> str:
    """Create a script that blocks specific package imports."""
    blocked_packages_str = ", ".join(f'"{pkg}"' for pkg in blocked_packages)
    
    return textwrap.dedent(f"""
        import sys
        import builtins
        from typing import Any

        # Store original import function
        original_import = builtins.__import__

        def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            '''Block imports of specific packages for testing.'''
            blocked_packages = [{blocked_packages_str}]
            
            # Check if this import should be blocked
            for blocked_pkg in blocked_packages:
                if name == blocked_pkg or name.startswith(blocked_pkg + "."):
                    raise ImportError(f"No module named '{{name}}' (blocked for testing)")
            
            # Allow all other imports
            return original_import(name, *args, **kwargs)

        # Install the import blocker
        builtins.__import__ = blocking_import
    """)


def create_integration_test_script(module_name: str, blocked_packages: List[str]) -> str:
    """Create a test script for a specific integration module."""
    return textwrap.dedent(f"""
        import sys
        import os
        from pathlib import Path
        
        # Add src directory to path
        src_path = Path.cwd() / "src"
        sys.path.insert(0, str(src_path))
        
        def test_integration_module():
            '''Test integration module with blocked dependencies.'''
            module_name = "{module_name}"
            blocked_packages = {blocked_packages}
            
            print(f"🧪 Testing {{module_name}} without {{blocked_packages}}...")
            
            try:
                # Try to import the integration module
                import_path = f"openlayer.lib.integrations.{{module_name}}"
                module = __import__(import_path, fromlist=[module_name])
                
                print(f"✅ Module {{module_name}} imported successfully")
                
                # Check for availability flag pattern
                availability_flags = [attr for attr in dir(module) 
                                    if attr.startswith('HAVE_') and 
                                    isinstance(getattr(module, attr), bool)]
                
                if availability_flags:
                    for flag in availability_flags:
                        flag_value = getattr(module, flag)
                        print(f"✅ Found availability flag: {{flag}} = {{flag_value}}")
                        if flag_value:
                            print(f"⚠️  WARNING: {{flag}} is True, but dependencies are blocked!")
                else:
                    print(f"⚠️  WARNING: No availability flag found (HAVE_* pattern)")
                
                # Try to find main integration classes (skip utility classes)
                integration_classes = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        attr.__module__ == module.__name__ and
                        not attr_name.startswith('_') and
                        # Skip utility classes that aren't integration points
                        not attr_name.endswith('Data') and
                        # Look for typical integration class patterns
                        ('Tracer' in attr_name or 'Processor' in attr_name or 'Callback' in attr_name)):
                        integration_classes.append((attr_name, attr))
                
                if not integration_classes:
                    print("⚠️  WARNING: No integration classes found")
                    return True
                
                # Test instantiation of integration classes
                for class_name, integration_class in integration_classes:
                    try:
                        print(f"🧪 Testing instantiation of {{class_name}}...")
                        instance = integration_class()
                        print(f"❌ FAIL: {{class_name}} instantiation should have failed without dependencies")
                        return False
                    except ImportError as e:
                        expected_keywords = ["required", "install", "pip install"]
                        error_msg = str(e).lower()
                        if any(keyword in error_msg for keyword in expected_keywords):
                            print(f"✅ {{class_name}} failed with helpful error: {{e}}")
                        else:
                            print(f"⚠️  {{class_name}} failed but error message could be more helpful: {{e}}")
                    except Exception as e:
                        print(f"❌ FAIL: {{class_name}} failed with unexpected error: {{e}}")
                        return False
                
                print(f"✅ All tests passed for {{module_name}}")
                return True
                
            except ImportError as e:
                print(f"❌ FAIL: Could not import {{module_name}}: {{e}}")
                return False
            except Exception as e:
                print(f"❌ FAIL: Unexpected error testing {{module_name}}: {{e}}")
                import traceback
                traceback.print_exc()
                return False
        
        if __name__ == "__main__":
            success = test_integration_module()
            sys.exit(0 if success else 1)
    """)


def run_integration_test(module_name: str, dependencies: List[str]) -> Tuple[bool, str]:
    """Run the integration test for a specific module."""
    # Create temporary files for the test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as blocker_file:
        blocker_file.write(create_import_blocker_script(dependencies))
        blocker_script = blocker_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as test_file:
        test_file.write(create_integration_test_script(module_name, dependencies))
        test_script = test_file.name
    
    try:
        # Run the test in a subprocess
        cmd = [
            sys.executable, 
            '-c', 
            f"exec(open('{blocker_script}').read()); exec(open('{test_script}').read())"
        ]
        
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        
        return result.returncode == 0, output
        
    except subprocess.TimeoutExpired:
        return False, "Test timed out"
    except Exception as e:
        return False, f"Test execution failed: {e}"
    finally:
        # Clean up temporary files
        try:
            Path(blocker_script).unlink()
            Path(test_script).unlink()
        except (FileNotFoundError, OSError):
            pass


class TestIntegrationConditionalImports:
    """Test class for integration conditional imports."""
    
    def test_all_integrations_handle_missing_dependencies(self) -> None:
        """Test that all integration modules handle missing dependencies correctly."""
        print("\n🚀 Testing all integration modules for conditional import handling...")
        
        failed_modules: List[str] = []
        all_results: List[Tuple[str, bool, str]] = []
        
        for module_name, dependencies in INTEGRATION_DEPENDENCIES.items():
            print(f"\n{'='*60}")
            print(f"Testing: {module_name}")
            print(f"Blocked dependencies: {dependencies}")
            print('='*60)
            
            success, output = run_integration_test(module_name, dependencies)
            
            print(output)
            
            if not success:
                failed_modules.append(module_name)
                print(f"❌ FAILED: {module_name}")
            else:
                print(f"✅ PASSED: {module_name}")
            
            all_results.append((module_name, success, output))
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        
        total_modules = len(INTEGRATION_DEPENDENCIES)
        passed_modules = total_modules - len(failed_modules)
        
        print(f"Total modules tested: {total_modules}")
        print(f"Passed: {passed_modules}")
        print(f"Failed: {len(failed_modules)}")
        
        if failed_modules:
            print(f"\nFailed modules: {', '.join(failed_modules)}")
            
            # Show details for failed modules
            for module_name, success, output in all_results:
                if not success:
                    print(f"\n--- {module_name} failure details ---")
                    print(output)
        
        # Assert all modules passed
        assert len(failed_modules) == 0, f"The following modules failed conditional import tests: {failed_modules}"
    
    def test_integration_modules_exist(self) -> None:
        """Test that all expected integration modules exist."""
        integrations_dir = Path("src/openlayer/lib/integrations")
        
        for module_name in INTEGRATION_DEPENDENCIES.keys():
            module_file = integrations_dir / f"{module_name}.py"
            assert module_file.exists(), f"Integration module {module_name}.py does not exist"
    
    def test_can_import_integrations_when_dependencies_available(self) -> None:
        """Test that integration modules can be imported when their dependencies are available."""
        print("\n🧪 Testing integration imports when dependencies are available...")
        
        # This test runs in the normal environment where dependencies may be available
        failed_imports: List[str] = []
        
        for module_name in INTEGRATION_DEPENDENCIES.keys():
            try:
                import_path = f"openlayer.lib.integrations.{module_name}"
                __import__(import_path)
                print(f"✅ {module_name} imported successfully")
            except ImportError as e:
                # This is expected if the dependency is not installed
                print(f"⚠️  {module_name} import failed (dependency not installed): {e}")
            except Exception as e:
                print(f"❌ {module_name} import failed with unexpected error: {e}")
                failed_imports.append(module_name)
        
        assert len(failed_imports) == 0, f"Unexpected import errors: {failed_imports}"


if __name__ == "__main__":
    # Run the tests when called directly
    test_instance = TestIntegrationConditionalImports()
    
    print("🧪 Running Integration Conditional Import Tests")
    print("=" * 60)
    
    try:
        test_instance.test_integration_modules_exist()
        print("✅ All integration modules exist")
        
        test_instance.test_can_import_integrations_when_dependencies_available()
        print("✅ Integration imports work when dependencies available")
        
        test_instance.test_all_integrations_handle_missing_dependencies()
        print("✅ All integration modules handle missing dependencies correctly")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        sys.exit(1) 