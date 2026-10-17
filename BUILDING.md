# Builds and GitHub Releases

The repository uses three separate release workflows, plus `validate.yml` for every push and pull request. A successful image build does not create a GitHub Release.

| Workflow | Result |
|---|---|
| `build-image.yml` | Builds and publishes the immutable candidate image named in `release.json`. |
| `promote-release.yml` | Publishes the stable image tag with the candidate's exact manifest digest. |
| `publish-release.yml` | Verifies the stable image, then creates the source tag and a published GitHub Release with the changelog and image digest. |

Validation runs check source and documentation. They do not publish an image or Release. Pushes to `main` that change an image input (`Containerfile`, `Containerfile.stack`, `release.json`, `stack.lock.json`, `patches/` or `build/`) start a candidate build; documentation and workflow changes do not rebuild an existing candidate. Published image and source tags are immutable.

GitHub-hosted CI caps the build at three compiler jobs (`MAX_JOBS=3`) through temporary copies of the two Containerfiles. The pinned sources, compiler targets and optimization flags stay the same, and the images copy and hash the tracked `Containerfile.stack`, so the stack identity is unchanged. Builders with more memory can use the tracked recipes as is.

## Image inputs

`Containerfile.stack` builds the stack image: FlashInfer with its cubin bundle, DeepGEMM and Triton, rebuilt from the locked sources and series on the vendor base. It is tagged by the hash of its inputs (`build/stack-key.sh`), so a candidate that changes only the SGLang series reuses it. `Containerfile` installs SGLang with its Rust extensions on top of that stack image, then runs the installed-image gates: the dependency check, the source and package identity check, and the CPU test list in `stack.lock.json`. `build/` holds the scripts both stages run; `scripts/` holds repository tooling that is not part of the image.

## Publish a new release

Run the release, documentation and source checks in [RUN.md](RUN.md), and complete hardware qualification before promoting a candidate. Wait for the candidate build to succeed and verify anonymous pull access. If a build fails, inspect its logs before retrying; confirm that the candidate tag is absent before starting another build of that tag.

Dispatch `promote-release.yml` from `main` and wait for success. Verify that the candidate and stable tags resolve to the same digest. Finalize and review the release documentation before selecting the source commit.

Dispatch the repository publisher with the full reviewed source SHA derived from Git:

```bash
git switch main
git pull --ff-only
release_target=$(git rev-parse HEAD)
gh workflow run publish-release.yml \
  --repo ormandj/sglang-deepseek-v41-flash-sm120 \
  --ref main \
  -f "release_target=$release_target"
```

It checks the source revision, candidate/stable digest equality and the canonical release body before creating the source tag and a non-draft, non-prerelease Release.

## Verify publication

```bash
release_tag=$(jq -er '.stable_tag' release.json)
gh release view "$release_tag" \
  --repo ormandj/sglang-deepseek-v41-flash-sm120 \
  --json url,tagName,isDraft,isPrerelease,body
gh api "repos/ormandj/sglang-deepseek-v41-flash-sm120/git/ref/tags/$release_tag" \
  --jq '.object'
```

Confirm that the Release is accessible, its body is the reviewed text and its tag resolves to the reviewed source commit. Independently confirm anonymous access to the stable image and that its digest matches the candidate.
