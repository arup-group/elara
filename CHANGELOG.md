# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.0]

## [0.1.0] - 2023-06-23

### Added

- Verbose option changed to 'debug' (#150), this is not a breaking change but old config files using `verbose = true` or some equivalent will now silently default to `debug = false` which is equivalent to the old `verbose = false`.
