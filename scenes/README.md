# Scene composition contract

The reviewed portfolio scene is a local OpenUSD composition. Its flattened or
generated USD is intentionally not committed because it contains external
NVIDIA assets and machine-specific reference paths.

Expected hierarchy:

```text
/World/AerospacePaintingCell
├── PaintBooth
├── LinearTrack
│   └── Carriage
│       └── PaintRobot
│           └── SprayTool
├── AircraftFixture
├── Workpiece
└── Process
    ├── TCPPath
    ├── SurfaceSamples
    ├── SprayCone
    └── CoverageOverlay
```

To reuse the composition locally:

1. Install NVIDIA Isaac Sim and resolve the official robot/environment assets.
2. Create a local base stage using generic, non-proprietary aircraft geometry.
3. Set `AEROSPACE_PAINTING_BASE_USD` to that stage.
4. Run `scripts/build_scene.py` with Isaac Sim's Python environment.

The checked-in Python package is portable and covers surface sampling, path
generation, spray-cone geometry, and geometric coverage evaluation. It does not
include an industrial robot controller post-processor or a paint-physics solver.
