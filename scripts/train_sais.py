"""Guard against treating the legacy RGB proxy as the SAIS comparator.

The manuscript comparison requires the official RGB-and-optical-flow recipe.
See docs/protocol.md for matched partitions, selection and aggregation.
"""

if __name__ == '__main__':
    raise SystemExit(
        'The legacy RGB-only proxy is not the manuscript SAIS comparator. '
        'Use the official RGB-and-optical-flow implementation; see docs/protocol.md.')
