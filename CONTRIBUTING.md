# Contributing to NFL Sentiment Analyzer

## Development Workflow

1. **Branch Structure**
   - `main`: Production-ready code
   - `dev`: Development branch for integrating new features
   - Feature branches: Create from `dev` for new features

2. **Creating a New Feature**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

3. **Making Changes**
   - Write clean, documented code
   - Follow PEP 8 style guide for Python code
   - Add tests for new features
   - Update documentation as needed

4. **Committing Changes**
   ```bash
   git add .
   git commit -m "feat: your descriptive commit message"
   ```
   
   Commit Message Prefixes:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Adding or modifying tests
   - `refactor:` Code refactoring
   - `style:` Code style changes

5. **Submitting Pull Requests**
   - Push your feature branch to GitHub
   - Create a PR against the `dev` branch
   - Provide a clear description of changes
   - Reference any related issues

6. **Code Review Process**
   - All PRs require at least one review
   - Address review comments
   - Keep PRs focused and reasonable in size

7. **Development Environment**
   - Use virtual environment
   - Keep dependencies updated
   - Test locally before pushing

## Setting Up Development Environment

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up pre-commit hooks (coming soon)

4. Copy environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your development credentials
   ```

## Testing

- Run tests before submitting PR:
  ```bash
  pytest  # Coming soon
  ```

## Documentation

- Update API documentation for new endpoints
- Keep README.md updated with new features
- Document any environment changes

## Questions?

Feel free to open an issue for:
- Feature proposals
- Bug reports
- Documentation improvements
- General questions

Thank you for contributing!
