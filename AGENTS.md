# Working in this repository

This is the public source and container repository for DeepSeek-V4.1-Flash on SM120. Documentation is written for users and contributors.

## Public content

README describes the current published image, requirements, installation, usage, measured behavior and actionable limitations. Link to the changelog and published Releases for history; do not add a candidate-by-candidate ledger or failed-build history. Release notes describe the changes actually delivered and their effect on users, with upgrade instructions and known limitations.

## Release integrity

Every reference to the image uses the complete release name, never an abbreviation or a floating tag. Published image and source tags are immutable; documentation corrections must not move tags or rebuild an existing candidate.

Bump `release.json` `candidate` (and the matching lock `candidate_tag` and image label) for any change to `Containerfile`, `Containerfile.stack`, `release.json`, `stack.lock.json`, `patches/` or `build/`. Those are the image inputs. A successful build alone does not qualify a release. Stable images and repository releases are created through the release workflows described in [BUILDING.md](BUILDING.md), with registry digest equality verified before publication.

`stack.lock.json` and `scripts/verify-patches.sh` define the official source commits, the hash-pinned patch series and the resulting trees. The vendor base supplies CUDA, PyTorch and runtime libraries only and does not establish source provenance.

## Checks

Run `./scripts/validate-release.sh` for image-input changes, `./scripts/validate-docs.sh` for documentation changes and `./scripts/verify-patches.sh` for changed source pins, patches or base image (it needs network access). For publication tooling changes, run `python3 scripts/test_validate_publish_release.py` and `python3 scripts/test_verify_release_target.py`.

## Measurements

Performance claims need measured evidence for the stated artifact, hardware, configuration and workload, and must say when a separately built image was measured. Report output tokens per second including reasoning with the concurrency and repetition count, and state that controlled measurements do not necessarily represent real-world performance.

## Contributions

Preserve contributor credit and upstream pull-request references in the patch tables. Do not hard-wrap public prose or use em dashes.
