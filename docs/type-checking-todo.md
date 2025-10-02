# Type Checking TODO

## Overview

The mypy type checking has been temporarily disabled in the CI pipeline due to 148+ type annotation errors across the codebase. This document outlines the work needed to re-enable strict type checking.

## Current Status

- **CI Status**: ✅ Mypy is commented out in `.github/workflows/ci.yml`
- **Error Count**: 148 errors across 27 files (down from 190 errors in 33 files)
- **Impact**: CI pipeline passes, but type safety is reduced

## Re-enabling Type Checking

To re-enable mypy in CI, uncomment these lines in `.github/workflows/ci.yml`:

```yaml
# - name: Type check with mypy
#   run: uv run mypy app --ignore-missing-imports --no-strict-optional --allow-untyped-calls --allow-untyped-defs --allow-incomplete-defs --allow-untyped-decorators
```

## Major Error Categories

### 1. Implicit Optional Types (Fixed with --no-strict-optional)
- **Issue**: PEP 484 prohibits implicit Optional types
- **Status**: ✅ Resolved with mypy flag
- **Example**: `def func(param: str = None)` should be `def func(param: str | None = None)`

### 2. Returning Any from Typed Functions
- **Files**: `app/models/nfl_sentiment_config.py`, `app/services/mlops/`, `app/core/`
- **Issue**: Functions declared to return specific types but return `Any`
- **Priority**: High - affects type safety

### 3. Pydantic Field Configuration
- **File**: `app/models/sentiment.py:118`
- **Issue**: `Field(...)` call doesn't match any overload variant
- **Priority**: Medium - model validation

### 4. Missing Type Annotations
- **Files**: Multiple service files
- **Issue**: Variables need explicit type hints
- **Example**: `requests = []` should be `requests: list[RequestType] = []`

### 5. Incompatible Type Assignments
- **Files**: `app/api/`, `app/services/`
- **Issue**: Assigning wrong types to variables
- **Priority**: High - runtime errors possible

## Recommended Approach

### Phase 1: Core Infrastructure (High Priority)
1. **Fix configuration and database modules**
   - `app/core/config.py` - Missing required arguments
   - `app/core/database.py` - Add function type annotations
   - `app/core/exceptions.py` - Fix Optional parameter types

2. **Fix model definitions**
   - `app/models/sentiment.py` - Fix Pydantic Field usage
   - `app/models/nfl_sentiment_config.py` - Fix return type annotations

### Phase 2: Service Layer (Medium Priority)
1. **MLOps services** - Fix return type annotations
2. **Data services** - Fix type assignments and annotations
3. **Caching service** - Fix async/await type issues

### Phase 3: API Layer (Lower Priority)
1. **API endpoints** - Fix response type annotations
2. **WebSocket handlers** - Add proper type hints
3. **Admin endpoints** - Fix complex type assignments

## Mypy Configuration Options

### Current Lenient Configuration
```bash
mypy app --ignore-missing-imports --no-strict-optional --allow-untyped-calls --allow-untyped-defs --allow-incomplete-defs --allow-untyped-decorators
```

### Progressive Strictness Levels

#### Level 1: Basic (Start Here)
```bash
mypy app --ignore-missing-imports --no-strict-optional
```

#### Level 2: Moderate
```bash
mypy app --ignore-missing-imports --no-strict-optional --disallow-untyped-defs
```

#### Level 3: Strict (Goal)
```bash
mypy app --strict
```

## File-by-File Progress Tracking

### ✅ Completed Files
- None yet

### 🔄 In Progress Files
- None yet

### ❌ Files Needing Work (148 errors)
- `app/models/nfl_sentiment_config.py` (4 errors)
- `app/models/sentiment.py` (1 error)
- `app/services/nfl_sentiment_engine.py` (3 errors)
- `app/core/config.py` (2 errors)
- `app/services/mlops/hopsworks_service.py` (1 error)
- `app/core/database.py` (2 notes)
- `app/services/database_migration_service.py` (4 errors)
- `app/services/data_ingestion_service.py` (3 errors)
- `app/services/data_archiving_service.py` (5 errors)
- `app/services/caching_service.py` (11 errors)
- `app/services/mlops/model_deployment_service.py` (4 errors)
- `app/core/openapi.py` (1 error)
- `app/core/monitoring.py` (7 errors)
- `app/services/data_processing_pipeline.py` (1 error)
- `app/services/analytics_service.py` (15 errors)
- `app/services/mlops/wandb_service.py` (7 errors)
- `app/services/mlops/huggingface_service.py` (4 errors)
- `app/core/rate_limiting.py` (2 errors)
- `app/core/dependencies.py` (1 error)
- `app/services/mlops/model_retraining_service.py` (7 errors)
- `app/core/middleware.py` (7 errors)
- `app/api/websocket.py` (4 errors)
- `app/api/sentiment.py` (25 errors)
- `app/api/data.py` (12 errors)
- `app/api/auth.py` (2 errors)
- `app/api/analytics.py` (3 errors)
- `app/services/mlops/mlops_service.py` (12 errors)
- `app/api/admin.py` (15 errors)

## Tools and Resources

### Mypy Configuration File
Consider creating a `mypy.ini` or `pyproject.toml` configuration:

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Start false, gradually enable
ignore_missing_imports = true
namespace_packages = true
explicit_package_bases = true

# Per-module configuration
[[tool.mypy.overrides]]
module = "app.core.*"
disallow_untyped_defs = true  # Enable for core modules first
```

### Useful Commands

```bash
# Check specific file
uv run mypy app/core/config.py

# Check with error codes
uv run mypy app --show-error-codes

# Generate type stubs for missing imports
uv run stubgen -p some_package

# Check only modified files
uv run mypy $(git diff --name-only --diff-filter=AM | grep '\.py$')
```

## Timeline Estimate

- **Phase 1**: 2-3 days (core infrastructure)
- **Phase 2**: 1-2 weeks (service layer)
- **Phase 3**: 1 week (API layer)
- **Total**: 2-3 weeks for full type safety

## Benefits of Completing This Work

1. **Runtime Error Prevention**: Catch type-related bugs before deployment
2. **Better IDE Support**: Improved autocomplete and refactoring
3. **Code Documentation**: Types serve as inline documentation
4. **Maintainability**: Easier to understand and modify code
5. **Team Productivity**: Reduced debugging time

## Next Steps

1. Create a GitHub issue to track this work
2. Start with Phase 1 files (core infrastructure)
3. Enable mypy for completed modules incrementally
4. Set up pre-commit hooks for new code
5. Consider using `mypy --strict` for new files only

---

**Note**: This is technical debt that should be prioritized based on development velocity and team capacity. The CI pipeline will continue to work without mypy, but type safety is compromised.