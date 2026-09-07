# Architecture

The project separates a portable process model from the local Isaac Sim scene.

```text
Generic curved surface
  -> sampled position + normal
  -> boustrophedon TCP path
  -> normal-aligned tool axis + stand-off
  -> spray-cone membership
  -> geometric exposure score
  -> coverage overlay and summary metrics

Local OpenUSD scene
  -> paint booth and fixture
  -> auxiliary linear rail
  -> 6-axis industrial robot
  -> spray-tool visualization
  -> animated process overlay
```

The portable Python modules run without Isaac Sim. The local visualization uses
Isaac Sim/OpenUSD to compose the robot, rail, booth, aircraft surface, cameras,
and process overlays in a shared scene.

## Motion sequence

1. The rail positions the robot at the first region.
2. The arm aligns the spray axis with the local surface normal.
3. The tool follows a constant-stand-off sweep while the cone is visible.
4. The arm steps to the adjacent row with spray disabled.
5. Rail and arm repeat the next sweep and progressively update coverage.
6. The tool retracts and the final geometric coverage map remains visible.
