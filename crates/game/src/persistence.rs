//! Saving and loading a game session to a compact binary blob.
//!
//! The world is infinite and deterministic from its seed, so we never store
//! generated terrain — only the seed plus the player's *edits* (modified chunk
//! storages) and session state (position, inventory, quest progress, time of
//! day). Reloading regenerates the world from the seed and overlays the edits.

use crate::Game;
use glam::Vec3;
use pixelcraft_core::chunk::ChunkStorage;
use pixelcraft_core::coords::ChunkPos;

const MAGIC: &[u8; 4] = b"PXC1";

struct Writer {
    buf: Vec<u8>,
}
impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.buf.extend_from_slice(b);
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.buf.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn i32(&mut self) -> Option<i32> {
        Some(i32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }
}

/// Serialise the full session to a byte buffer.
pub fn save_to_bytes(game: &Game) -> Vec<u8> {
    let mut w = Writer::new();
    w.buf.extend_from_slice(MAGIC);
    w.u64(game.manager.seed());
    w.f32(game.environment.time_of_day);
    w.f32(game.time);

    // Player.
    let p = &game.player;
    w.f32(p.position.x);
    w.f32(p.position.y);
    w.f32(p.position.z);
    w.f32(p.yaw);
    w.f32(p.pitch);

    // Inventory: selected slot + count per block id.
    w.u32(game.inventory.selected as u32);
    let reg_len = game.manager.registry.len();
    w.u32(reg_len as u32);
    for id in 0..reg_len {
        w.u32(game.inventory.count(pixelcraft_core::block::BlockId(id as u16)));
    }

    // Quest progress.
    w.u32(game.quests.completed_count() as u32);
    w.u32(game.quests.progress());

    // Edits.
    let edits = game.manager.collect_edits();
    w.u32(edits.len() as u32);
    for (pos, storage) in &edits {
        w.i32(pos.x);
        w.i32(pos.y);
        w.i32(pos.z);
        w.bytes(&storage.to_bytes());
    }
    w.buf
}

/// Reconstruct a session from [`save_to_bytes`]. `worker_count` controls the
/// rebuilt streaming pool. Returns `None` if the blob is malformed.
pub fn load_from_bytes(bytes: &[u8], worker_count: usize) -> Option<Game> {
    let mut r = Reader::new(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let seed = r.u64()?;
    let time_of_day = r.f32()?;
    let time = r.f32()?;

    // Build a fresh game on the same seed, then overlay saved state.
    let mut game = Game::new(seed, crate::StreamConfig::default(), worker_count);
    game.environment.time_of_day = time_of_day;
    game.time = time;

    let px = r.f32()?;
    let py = r.f32()?;
    let pz = r.f32()?;
    game.player.position = Vec3::new(px, py, pz);
    game.player.yaw = r.f32()?;
    game.player.pitch = r.f32()?;

    let selected = r.u32()? as usize;
    let reg_len = r.u32()? as usize;
    for id in 0..reg_len {
        let count = r.u32()?;
        if count > 0 {
            game.inventory
                .add(pixelcraft_core::block::BlockId(id as u16), count);
        }
    }
    game.inventory.selected = selected % crate::HOTBAR.len();

    let completed = r.u32()?;
    let progress = r.u32()?;
    game.quests.restore(completed as usize, progress);

    let edit_count = r.u32()?;
    let mut edits = Vec::with_capacity(edit_count as usize);
    for _ in 0..edit_count {
        let x = r.i32()?;
        let y = r.i32()?;
        let z = r.i32()?;
        let sbytes = r.bytes()?;
        let (storage, _) = ChunkStorage::from_bytes(sbytes)?;
        edits.push((ChunkPos::new(x, y, z), storage));
    }
    game.manager.restore_edits(edits);

    Some(game)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamConfig;
    use glam::Vec3;
    use pixelcraft_core::block::blocks;
    use pixelcraft_core::coords::BlockPos;
    use pixelcraft_physics::MovementInput;

    fn test_config() -> StreamConfig {
        StreamConfig {
            view_distance: 2,
            min_chunk_y: 0,
            max_chunk_y: 4,
            unload_margin: 1,
            max_uploads_per_update: 4096,
            max_meshes_per_update: 4096,
            max_jobs_per_update: 4096,
        }
    }

    #[test]
    fn save_load_roundtrip_preserves_state() {
        let mut game = Game::new(2024, test_config(), 0);
        // Settle, then make some edits and changes.
        for _ in 0..40 {
            game.update(MovementInput::default(), 1.0 / 60.0);
            if game.manager.is_settled() {
                break;
            }
        }
        let edit_pos = BlockPos::new(
            game.player.position.x as i32,
            80,
            game.player.position.z as i32,
        );
        game.manager.set_block(edit_pos, blocks::LANTERN);
        game.inventory.add(blocks::PLANK, 7);
        game.inventory.select(3);
        game.player.position = Vec3::new(12.5, 70.0, -4.5);
        game.environment.time_of_day = 0.42;

        let blob = save_to_bytes(&game);
        let loaded = load_from_bytes(&blob, 0).expect("load");

        assert_eq!(loaded.manager.seed(), 2024);
        assert!((loaded.player.position.x - 12.5).abs() < 1e-4);
        assert!((loaded.environment.time_of_day - 0.42).abs() < 1e-5);
        assert_eq!(loaded.inventory.count(blocks::PLANK), 7);
        assert_eq!(loaded.inventory.selected, 3);

        // The edited block must reappear once its chunk streams back in.
        let mut loaded = loaded;
        loaded.player.position = Vec3::new(edit_pos.x as f32, 75.0, edit_pos.z as f32);
        for _ in 0..40 {
            loaded.update(MovementInput::default(), 1.0 / 60.0);
            if loaded.manager.is_settled() {
                break;
            }
        }
        assert_eq!(loaded.manager.world.block_at(edit_pos), blocks::LANTERN);
    }

    #[test]
    fn load_rejects_bad_data() {
        assert!(load_from_bytes(b"nope", 0).is_none());
        assert!(load_from_bytes(&[], 0).is_none());
    }
}
