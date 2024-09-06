# Changelog
## [0.4.16] - 2024-09-06
### Fixed
- fix a bug that bool is captured as int64_t in inliner.
- fix a bug of additional vars in inliner.

## [0.4.15] - 2024-08-14
### Added
- add compiled conversion to greatly reduce launch overhead of inline function. currently only inline cuda/metal kernels in cumm support this.

## [0.4.14] - 2024-07-31
### Fixed
- fix a bug in simple type analysis
### Added 
- add metal constant filter in type analysis
- add numpy scalar type support in type analysis

## [0.4.13] - 2024-07-27
### Added
- add attributes and `code_before_func_def` support for apple metal

## [0.4.12] - 2024-01-03
### Added
- add support for param class reload by check id in `sys.module`

## [0.4.11] - 2023-11-11
### Fixed 
* fix wrong version.txt


## [0.4.10] - 2023-11-07
### Fixed 
* fix wrong enum typing

## [0.4.9] - 2023-10-17
### Fixed 
* fix missing includes and alias when use impl-only dep with header only

## [0.4.8] - 2023-07-05
### Added 
* add extern c to cuda globals

## [0.4.7] - 2023-04-07
### Added 
* Add function for inliner code inspect
* add a option to faster check whether perform reload
### Fixed 
* fix pre-capture small bug

## [0.4.6] - 2023-02-06
### Added 
* Add support for complex inliner and cpu inliner

## [0.4.5] - 2023-02-01
### Added 
* Add better capture support for inliner.

## [0.4.4] - 2022-11-09
### Added
* Add PCCM_DISABLE_CODE_CHANGE to disable auto code override for debugging.

## [0.4.3] - 2022-10-13
### Added
* Add include path for pybind only code
### Fixed
* Fix some compiler don't support '~'.

## [0.4.2] - 2022-09-25
### Fixed
* Fix dynamic decl problem.

## [0.4.1] - 2022-09-25
### Fixed
* fix small bug in annotation gen.

## [0.3.5] - 2022-07-20
### Added
* add gen_cmake.
### Changed
* BREAK CHANGE: Change build_pybind api. 
### Fixed
* fix some bugs.
