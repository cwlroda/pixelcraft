# PixelCraft — contributor guide

A cosy, performance-focused 3D voxel game in Rust where you play a sentient cat.
Custom engine (no game framework); rendering on `wgpu`, windowing on `winit`.

## Workspace layout

```
crates/core      pixelcraft-core      coords, block registry, palette-compressed
                                      chunk storage, sparse World (no deps on render)
crates/worldgen  pixelcraft-worldgen  seedable Perlin/fBm, biomes, deterministic
                                      terrain + caves + ores + decoration + structures
crates/mesh      pixelcraft-mesh      greedy meshing + ambient occlusion + cross
                                      plants; produces Vertex { pos, normal, color
                                      (luminance), uv, layer }
crates/physics   pixelcraft-physics   AABB swept voxel collision, cat controller
crates/game      pixelcraft-game      streaming, raycast, camera, inventory,
                                      interaction, entities, quests, crafting,
                                      particles, persistence, environment, render/
```

`crates/game/src/render/` is gated behind the `render` (windowed) and `capture`
(headless PNG/GIF) features so the default build/test stays GPU-free.

## Key design notes

- **Chunks** are 32³, power-of-two for shift/mask coord math. Storage is a
  paletted container: uniform chunks use zero per-voxel memory.
- **Meshing** runs in parallel across threads (scoped pool sharing `&World`,
  which is `Sync`) within a per-update budget. AO is baked into the merge key.
- **Textures** are procedural (one tile per block) in a `texture_2d_array`;
  `uv` is in block units so tiles repeat across greedy-merged quads.
- **Streaming**: `ChunkManager` generates on a worker pool, meshes dirty chunks,
  unloads with hysteresis, and keeps player edits in an `edits` cache so they
  survive unload (and feed `persistence`).
- The whole simulation is headless/testable; `Game` (lib.rs) ties it together
  and is driven by `render/app.rs` (window) or `bin/screenshot.rs` (capture).

## Build / test / run

```bash
cargo test                                              # ~120 headless tests
cargo run --release                                     # headless self-check sim
cargo run --bin pixelcraft --features render -- [seed]  # windowed game (GPU+display)
cargo run --bin screenshot --features capture -- [seed] [dir]   # PNGs + day-cycle GIF
```

The capture path renders offscreen via software Vulkan (Mesa **lavapipe**); in a
fresh container install it with
`apt-get install -y mesa-vulkan-drivers` (the loader alone is not enough).

## Conventions

- Keep core/worldgen/mesh/physics free of rendering deps.
- New blocks: add to `core::block` (id const + registry entry), give them a
  texture pattern in `render/textures.rs`, and a mesh `RenderKind`.
- Add a headless test for new simulation logic; verify visuals with the
  screenshot tool.
