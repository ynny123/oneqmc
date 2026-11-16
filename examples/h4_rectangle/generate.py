"""Generate the six H4 rectangles from the public Globe hydrogen experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
ANGLES_DEG = (85.0, 86.0, 87.0, 88.0, 89.0, 89.999)
RADIUS_BOHR = 3.2843
SOURCE_COMMIT = '942feba753a0ab912c23fdebc873ff9aefeb49f6'
SOURCE_BASE = f'https://github.com/n-gao/globe/blob/{SOURCE_COMMIT}'


def molecule(theta):
    x = RADIUS_BOHR * math.cos(math.radians(theta / 2))
    y = RADIUS_BOHR * math.sin(math.radians(theta / 2))
    return {
        'coords': [[x, y, 0.0], [x, -y, 0.0], [-x, y, 0.0], [-x, -y, 0.0]],
        'charges': [1, 1, 1, 1], 'charge': 0, 'spin': 0, 'unit': 'bohr',
        'extras': {'theta_deg': theta, 'radius_bohr': RADIUS_BOHR,
                   'geometry_source': SOURCE_BASE + '/globe/systems/__init__.py#L29'},
    }


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for index, theta in enumerate(ANGLES_DEG):
        path = output / f'{index:02d}.yaml'
        content = yaml.safe_dump(molecule(theta), sort_keys=False)
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f'Refusing to replace different data: {path}')
        path.write_text(content)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO / 'data/h4_globe6')
    args = parser.parse_args()
    print(json.dumps({'output': str(args.output.resolve()), 'unit': 'bohr',
                      'radius_bohr': RADIUS_BOHR, 'angles_deg': ANGLES_DEG,
                      'source_commit': SOURCE_COMMIT,
                      'sha256': generate(args.output)}, indent=2))


if __name__ == '__main__':
    main()
