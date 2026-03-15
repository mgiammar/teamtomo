# Release Instructions

## Prerequisites
Ensure you have push access to the repository and are on an up-to-date main branch:

```bash
git checkout main
git pull upstream main
```

## Single Package Release

To release a single package, tag it with the pattern `package-name@vX.Y.Z`:

```bash
git tag package-name@v3.4.5
git push upstream main --follow-tags
```

**What happens next:**

1. The `Deploy` workflow triggers automatically
2. CI verification ensures tests passed on main
3. The package is built and published to PyPI
4. A GitHub Release is created with the built artifacts

## Coordinated Release (All Packages)

To release all packages in the workspace with the same version, tag the commit with `teamtomo@vX.Y.Z`:

```bash
git tag teamtomo@v3.4.5
git push upstream main --follow-tags
```

**What happens next:**

1. The `Coordinate Release` workflow triggers
2. Waits for CI to pass on Python 3.13
3. Extracts the version from the tag
4. Automatically tags all workspace packages with `package-name@v1.0.0`
5. Individual `Deploy` workflows trigger for each package
6. All packages are built and published to PyPI simultaneously

**Note:** The coordinated release approach is recommended when you want to ensure version consistency across all packages in the TeamTomo workspace.
