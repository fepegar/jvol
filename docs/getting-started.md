## Installation

```shell
pip install git+git://github.com/fepegar/jvol.git
```

## Usage

### Python

### Shell

```shell
$ jvol --help

 Usage: jvol [OPTIONS] INPUT_PATH OUTPUT_PATH

 Tool for converting medical images to and from JPEG-encoded volumes.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────╮
│ *    input_path       FILE  [default: None] [required]                               │
│ *    output_path      FILE  [default: None] [required]                               │
╰──────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────╮
│ --quality             -q      INTEGER RANGE [1<=x<=100]  [default: 60]               │
│ --block-size          -b      INTEGER RANGE [x>=2]       [default: 8]                │
│ --verbose             -v      INTEGER                    [default: 0]                │
│ --help                                                   Show this message and exit. │
╰──────────────────────────────────────────────────────────────────────────────────────╯
```
