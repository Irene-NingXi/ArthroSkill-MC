# Paper release and archive mapping

The `v1.0-paper` GitHub release identifies one code-and-data snapshot. Its tag resolves to the exact Git commit; the release's `release-manifest.json` records that commit, source-data manifest checksum, package lock checksum and manuscript checksum. Release immutability is enabled before publication so the tag and attached assets cannot subsequently be replaced.

| Component | Identifier |
| --- | --- |
| Code / GitHub Release | `v1.0-paper` |
| Public source-data release | `v5-public-1` |
| Supplied data workbook SHA-256 | `ae9516771048394df8acf2ee1dcba19cf0eaa335fbfc875e12a4c48da18a7c19` |
| Supplied MedIA manuscript SHA-256 | `98d0a211580880dca7cd0a1cdc60e74e8f23ba63e62adc5e44c214f5c5f80f46` |
| Complete table dictionary and checksums | `data/source_data/data_dictionary.json` and `data/source_data/manifest.json` |
| Recomputed statistical reference | `docs/source_data_summary.json` |
| Python and dependency profile | `.python-version` and `requirements-lock.txt` |
| Citation metadata | `CITATION.cff` |
| Zenodo version DOI | [10.5281/zenodo.23692599](https://doi.org/10.5281/zenodo.23692599) |
| Zenodo concept DOI (all versions) | [10.5281/zenodo.23692598](https://doi.org/10.5281/zenodo.23692598) |

The release packages only files tracked at its tag. It includes current public CSV tables and reproducibility code. Restricted recordings, the private workbook container, the manuscript document itself, fitted study weights, external SAIS source and generated synthetic checkpoints are excluded. The data and manuscript checksums establish which supplied inputs correspond to the release; they do not publish those private files.

The source-data values are recorded predictions, not results of a new full vision-model fitting run. The Quickstart checks their complete numerical reanalysis. The separate synthetic smoke test verifies the official RGB/flow SAIS fitting and checkpoint-evaluation pathway without using clinical videos or claiming new study performance.

## Zenodo archiving

Zenodo archiving requires the repository owner's Zenodo account and GitHub integration. The public GitHub release alone does not mint a DOI. The `v1.0-paper` record is published at [Zenodo record 23692599](https://zenodo.org/records/23692599), linked to commit `92d3286e77987d042b16a7a3739af40910e0ddd5`. The version DOI identifies this exact snapshot; the concept DOI identifies the version family. The DOI was assigned after the immutable GitHub release was published. Its original attached manifest therefore records the DOI as unassigned at release time; the frozen tag and assets remain unchanged.

For future versions, use the [official GitHub-Zenodo integration](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content):

1. Keep automatic archiving enabled for `Irene-NingXi/ArthroSkill-MC`.
2. Prepare and validate the next version before creating its GitHub release.
3. Wait for Zenodo to process the new release, then verify its version DOI and archived files.
4. Review the creator list, title, version and separate code/data reuse terms.
5. Add the assigned DOI to citation metadata without changing previously published immutable tags or assets.

Software is MIT-licensed. Public source-data access and reuse follow `data/source_data/TERMS.md`; this archive does not assign a new CC BY or CC0 data licence. Third-party SAIS code and weights retain their own terms and are not included. An archive licence must not silently relicense participant data or third-party material.
