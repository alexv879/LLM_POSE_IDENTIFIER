# Contributing to Pose LLM Identifier

Thank you for considering contributing! This project implements state-of-the-art pose estimation research and welcomes improvements.

## How to Contribute

### Reporting Bugs

Found a bug? Please [open an issue](https://github.com/alexv879/LLM_POSE_IDENTIFIER/issues) with:
- **Description**: What went wrong?
- **Steps to reproduce**: How can we see the bug?
- **Expected behavior**: What should happen?
- **Environment**: OS, Python version, GPU model
- **Error messages**: Full error traceback

### Suggesting Enhancements

Have an idea? [Open an issue](https://github.com/alexv879/LLM_POSE_IDENTIFIER/issues) with:
- **Use case**: Why is this useful?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches?

### Pull Requests

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature`
3. **Make changes**: Follow code style (PEP 8)
4. **Test**: Ensure code works
5. **Commit**: `git commit -m "Add feature X"`
6. **Push**: `git push origin feature/your-feature`
7. **Open PR**: Describe your changes

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/LLM_POSE_IDENTIFIER.git
cd LLM_POSE_IDENTIFIER

# Create dev environment
python -m venv dev_env
source dev_env/bin/activate  # Windows: dev_env\Scripts\activate

# Install in editable mode
pip install -e .

# Install dev dependencies
pip install pytest black flake8 mypy
```

### Code Style

- **Python**: Follow PEP 8
- **Docstrings**: Google style
- **Type hints**: Use where appropriate
- **Comments**: Explain "why", not "what"

### Testing

```bash
# Run tests
pytest tests/

# Check code style
black --check .
flake8 .
```

### Documentation

- Update README.md if adding features
- Add docstrings to new functions/classes
- Update configs/ if changing parameters

## Areas for Contribution

### Priority

- [ ] Stage 3 ensemble implementation
- [ ] Stage 4 autoencoder refinement
- [ ] Stage 5 LLM interpretability
- [ ] Better data augmentation strategies
- [ ] Multi-GPU training support

### Nice to Have

- [ ] 3D pose extension
- [ ] Video/temporal tracking
- [ ] Web demo interface
- [ ] Docker containerization
- [ ] More dataset loaders (MPII, etc.)
- [ ] Improved visualization tools

### Documentation

- [ ] Video tutorials
- [ ] More examples
- [ ] FAQ section
- [ ] Troubleshooting guide

## Questions?

- **Discussions**: [GitHub Discussions](https://github.com/alexv879/LLM_POSE_IDENTIFIER/discussions)
- **Issues**: [GitHub Issues](https://github.com/alexv879/LLM_POSE_IDENTIFIER/issues)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
