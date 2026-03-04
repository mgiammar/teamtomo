# Migration Progress

Tracking migration progress of TeamTomo packages into the monorepo and the depreciation/archiving of old GitHub repositories.

## Migration into the Monorepo

Which packages have been added into the monorepo thus far. Please follow [Migration Guide for an Existing Package](./migrate-existing-repo.md) for instructions on adding a pre-existing package into the monorepo.

_Note: File i/o packages remain separate from the monorepo_

### Primitives packages migration

- [x] `torch-fourier-slice`             - extracting/inserting central slices of Fourier transforms
- [x] `torch-fourier-rescale`           - rescale by padding/cropping Fourier transforms
- [x] `torch-fourier-shift`             - subpixel shift by phase shifting Fourier transforms
- [x] `torch-ctf`                       - Contrast Transfer Function utilities
- [x] `torch-fourier-filter`            - Fourier space filters
- [x] `torch-fourier-shell-correlation` - correlation as a function of spatial frequency
- [x] `torch-image-interpolation`       - sample values from or insert values into images
- [x] `torch-transform-image`           - affine transforms of images
- [x] `torch-cubic-spline-grids`        - continuous parametrisations of 1-4D spaces
- [x] `torch-subpixel-crop`             - crop from images with subpixel precision
- [x] `torch-find-peaks`                - find and refine peaks in images
- [x] `torch-grid-utils`                - coordinate grids, frequency grids and shape generation
- [x] `torch-so3`                       - 3D rotation operations and utilities
- [x] `torch-affine-utils`              - affine matrix generation for 2D/3D coordinates
- [x] `torch-tomogram`                  - tomogram data interface (renamed to torch-tilt-series during migration)

### Algorithms packages migration

- [ ] `torch-2dtm`                        - 2D template matching in cryo-EM images
- [ ] `torch-tiltxcorr`                   - coarse tilt series alignment for cryo-ET tilt series
- [ ] `torch-refine-tilt-axis-angle`      - tilt axis angle refinement for cryo-ET tilt series
- [ ] `torch-cryoeraser`                  - erase regions in cryo-EM images
- [ ] `torch-segment-fiducials-2d`        - segment gold fiducials in cryo-EM images
- [ ] `torch-segment-tomogram-boundaries` - detect boundaries of cryo-ET volumes
- [ ] `torch-motion-correction`           - correct local motion in cryo-EM images
- [ ] `torch-ctf-estimation`              - estimate local defocus in cryo-EM images

### Util packages migration

- [ ] `teamtomo-basemodel` - Helpful Pydantic wrapper for parsing, validation, and serialization reuse.

## Archiving old repositories

Which package repositories have been tagged as depreciated and archived. Document on how to archive an old repository in progress.

### Primitives packages archiving

- [ ] `torch-fourier-slice`
- [ ] `torch-fourier-rescale`
- [ ] `torch-fourier-shift`
- [ ] `torch-ctf`
- [ ] `torch-fourier-filter`
- [ ] `torch-fourier-shell-correlation`
- [ ] `torch-image-interpolation`
- [ ] `torch-transform-image`
- [ ] `torch-cubic-spline-grids`
- [ ] `torch-subpixel-crop`
- [ ] `torch-find-peaks`
- [ ] `torch-grid-utils`
- [ ] `torch-so3`
- [ ] `torch-affine-utils`
- [ ] `torch-tomogram`

### Algorithms packages archiving

- [ ] `torch-2dtm`
- [ ] `torch-tiltxcorr`
- [ ] `torch-refine-tilt-axis-angle`
- [ ] `torch-cryoeraser`
- [ ] `torch-segment-fiducials-2d`
- [ ] `torch-segment-tomogram-boundaries`
- [ ] `torch-motion-correction`
- [ ] `torch-ctf-estimation`

### Utils packages archiving

- [ ] `teamtomo-basemodel`
