This package contains DiffOrb's optional data manager, data manifest, and
DiffOrb-maintained data files that are bundled with the package.

External runtime data files are installed into DiffOrb's platform data
directory, not tracked in the public source repository. Installed files take
precedence over bundled package data. Run `python -m difforb.data dir` to show
the active data directory.
