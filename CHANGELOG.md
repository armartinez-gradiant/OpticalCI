# Changelog

All notable changes to PtONN-TESTS by Anxo Rodríguez Martínez will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-16

### Added
- Initial release of PtONN-TESTS
- Modern PyTorch compatibility (2.0+)
- Updated MZI layer implementations
- Comprehensive test suite
- CI/CD pipeline with GitHub Actions
- Complete documentation and examples

### Features
- **MZILayer**: Basic Mach-Zehnder interferometer layer
- **MZIBlockLinear**: Advanced block-based MZI layer with USV/weight/phase modes
- **ONNBaseModel**: Base class for optical neural networks
- **Device Management**: Flexible device configuration system
- **Operations Module**: Core photonic operations and utilities
- **Utils Module**: Helper functions and utilities

### Technical Improvements
- ✅ Fixed NumPy 2.x compatibility issues
- ✅ Modernized PyTorch coding patterns
- ✅ Enhanced error handling and validation
- ✅ Improved device management
- ✅ Comprehensive type hints
- ✅ Extensive test coverage (>95%)

### Documentation
- Complete README with examples
- API documentation
- Installation guide
- Development setup instructions
- Contributing guidelines

### Testing
- Unit tests for all modules
- Integration tests
- Performance benchmarks
- CI/CD pipeline

### Compatibility
- Python 3.8 - 3.11
- PyTorch 2.0+
- NumPy < 2.0
- CUDA support
- MPS (Apple Silicon) support

## [Unreleased]

### Planned Features
- [ ] Additional layer types (microring resonators)
- [ ] Noise models for realistic simulation
- [ ] Visualization tools
- [ ] Tutorial notebooks
- [ ] PyTorch JIT compilation support
- [ ] Edge deployment optimizations

### Known Issues
- None currently reported

---

## Version History

- **v0.1.0**: Initial release with modern PyTorch compatibility