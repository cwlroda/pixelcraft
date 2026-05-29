//! Chunk streaming: deciding which chunks should be resident around the player,
//! generating them on a background worker pool, meshing dirty chunks within a
//! per-frame budget, and unloading distant chunks.
//!
//! Generation and meshing are the two heaviest CPU costs in a voxel engine, so
//! both are rate-limited per update to keep frame times smooth (no hitches when
//! crossing chunk borders), and generation is offloaded to worker threads so the
//! main thread never blocks on terrain noise.

use ahash::{AHashMap, AHashSet};
use pixelcraft_core::block::{BlockId, BlockRegistry};
use pixelcraft_core::chunk::ChunkStorage;
use pixelcraft_core::coords::{BlockPos, ChunkPos};
use pixelcraft_core::world::World;
use pixelcraft_mesh::{mesh_chunk, ChunkMesh};
use pixelcraft_worldgen::WorldGenerator;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::sampler::ChunkNeighborSampler;

/// Shared work queue feeding the generation workers. A condvar lets workers
/// sleep while idle instead of spinning.
struct JobQueue {
    inner: Mutex<JobInner>,
    signal: Condvar,
}

struct JobInner {
    queue: std::collections::VecDeque<ChunkPos>,
    shutdown: bool,
}

impl JobQueue {
    fn new() -> Self {
        Self {
            inner: Mutex::new(JobInner {
                queue: std::collections::VecDeque::new(),
                shutdown: false,
            }),
            signal: Condvar::new(),
        }
    }

    fn push(&self, pos: ChunkPos) {
        let mut g = self.inner.lock().unwrap();
        g.queue.push_back(pos);
        drop(g);
        self.signal.notify_one();
    }

    /// Block until a job is available or shutdown is signalled.
    fn pop(&self) -> Option<ChunkPos> {
        let mut g = self.inner.lock().unwrap();
        loop {
            if let Some(pos) = g.queue.pop_front() {
                return Some(pos);
            }
            if g.shutdown {
                return None;
            }
            g = self.signal.wait(g).unwrap();
        }
    }

    fn shutdown(&self) {
        let mut g = self.inner.lock().unwrap();
        g.shutdown = true;
        drop(g);
        self.signal.notify_all();
    }
}

/// Tunable streaming limits.
#[derive(Clone, Copy)]
pub struct StreamConfig {
    /// Horizontal radius, in chunks, of the loaded region.
    pub view_distance: i32,
    /// Inclusive chunk-Y range that bounds the (finite-height) world column.
    pub min_chunk_y: i32,
    pub max_chunk_y: i32,
    /// Extra chunks kept loaded beyond `view_distance` before unloading, to
    /// avoid thrashing when the player jitters across a border (hysteresis).
    pub unload_margin: i32,
    /// Max generated chunks integrated into the world per update.
    pub max_uploads_per_update: usize,
    /// Max chunks (re)meshed per update.
    pub max_meshes_per_update: usize,
    /// Max new generation jobs submitted per update.
    pub max_jobs_per_update: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            view_distance: 10,
            min_chunk_y: 0,
            max_chunk_y: 5,
            unload_margin: 2,
            max_uploads_per_update: 8,
            max_meshes_per_update: 6,
            max_jobs_per_update: 24,
        }
    }
}

/// Streaming statistics for the HUD / profiling.
#[derive(Clone, Copy, Default, Debug)]
pub struct StreamStats {
    pub loaded_chunks: usize,
    pub meshed_chunks: usize,
    pub inflight: usize,
    pub voxel_memory_bytes: usize,
}

pub struct ChunkManager {
    pub world: World,
    pub registry: Arc<BlockRegistry>,
    pub config: StreamConfig,
    gen: WorldGenerator,
    meshes: AHashMap<ChunkPos, ChunkMesh>,
    /// Bumped every time a chunk is (re)meshed, so the renderer can detect when
    /// a cached GPU buffer is stale (e.g. after the player edits a block).
    mesh_versions: AHashMap<ChunkPos, u64>,
    /// Player-edited chunk storages, retained across unload so edits persist
    /// (and can be saved). Re-applied when a chunk streams back in.
    edits: AHashMap<ChunkPos, ChunkStorage>,
    inflight: AHashSet<ChunkPos>,
    result_tx: Sender<(ChunkPos, ChunkStorage)>,
    result_rx: Receiver<(ChunkPos, ChunkStorage)>,
    queue: Arc<JobQueue>,
    workers: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    /// Threads used to mesh the per-update batch in parallel (1 = sequential).
    mesh_threads: usize,
    last_center: Option<ChunkPos>,
}

impl ChunkManager {
    /// Create a manager. `worker_count == 0` runs generation synchronously on
    /// the calling thread (deterministic, used in tests); otherwise that many
    /// background threads generate terrain.
    pub fn new(seed: u64, config: StreamConfig, worker_count: usize) -> Self {
        let registry = Arc::new(BlockRegistry::with_defaults());
        let gen = WorldGenerator::new(seed);
        let (result_tx, result_rx) = channel();
        let queue = Arc::new(JobQueue::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut workers = Vec::new();
        for _ in 0..worker_count {
            let queue = Arc::clone(&queue);
            let tx = result_tx.clone();
            let gen = gen.clone();
            workers.push(std::thread::spawn(move || {
                while let Some(pos) = queue.pop() {
                    let storage = gen.generate_chunk(pos);
                    // If the receiver is gone the manager is shutting down.
                    if tx.send((pos, storage)).is_err() {
                        break;
                    }
                }
            }));
        }

        Self {
            world: World::new(),
            registry,
            config,
            gen,
            meshes: AHashMap::new(),
            mesh_versions: AHashMap::new(),
            edits: AHashMap::new(),
            inflight: AHashSet::new(),
            result_tx,
            result_rx,
            queue,
            workers,
            shutdown,
            // Mesh in parallel when generation workers were requested; the sync
            // path (worker_count == 0) meshes on the calling thread for
            // deterministic tests.
            mesh_threads: worker_count.max(1),
            last_center: None,
        }
    }

    fn synchronous(&self) -> bool {
        self.workers.is_empty()
    }

    /// The world seed this manager generates from.
    pub fn seed(&self) -> u64 {
        self.gen.seed()
    }

    /// Terrain surface height for a world column (delegates to the generator),
    /// without needing the chunk to be loaded.
    pub fn surface_height(&self, wx: i32, wz: i32) -> i32 {
        self.gen.surface_height(wx, wz)
    }

    /// Biome at a world column.
    pub fn biome_at(&self, wx: i32, wz: i32) -> pixelcraft_worldgen::Biome {
        self.gen.biome_at(wx, wz)
    }

    /// Chunk containing a world position.
    pub fn chunk_of(pos: glam::Vec3) -> ChunkPos {
        BlockPos::new(
            pos.x.floor() as i32,
            pos.y.floor() as i32,
            pos.z.floor() as i32,
        )
        .chunk()
    }

    /// Drive streaming for a player at `center` (world coordinates).
    pub fn update(&mut self, center: glam::Vec3) {
        let cc = Self::chunk_of(center);
        self.submit_missing(cc);
        self.integrate_results(cc);
        self.unload_distant(cc);
        self.remesh_dirty(cc);
        self.last_center = Some(cc);
    }

    /// Submit generation jobs for desired-but-missing chunks, nearest first.
    fn submit_missing(&mut self, center: ChunkPos) {
        let vd = self.config.view_distance;
        let mut wanted: Vec<ChunkPos> = Vec::new();
        for dz in -vd..=vd {
            for dx in -vd..=vd {
                if dx * dx + dz * dz > vd * vd {
                    continue;
                }
                for cy in self.config.min_chunk_y..=self.config.max_chunk_y {
                    let p = ChunkPos::new(center.x + dx, cy, center.z + dz);
                    if !self.world.contains_chunk(p) && !self.inflight.contains(&p) {
                        wanted.push(p);
                    }
                }
            }
        }
        // Prioritise the chunks closest to the player.
        wanted.sort_by_key(|p| p.distance_sq(center));

        let budget = self.config.max_jobs_per_update;
        for p in wanted.into_iter().take(budget) {
            // A previously edited chunk is restored from the edit store rather
            // than regenerated, so player changes persist across streaming.
            if let Some(storage) = self.edits.get(&p) {
                self.world.insert_chunk(p, storage.clone());
                if let Some(c) = self.world.get_chunk_mut(p) {
                    c.modified = true;
                }
                self.mark_dirty_with_neighbours(p);
                continue;
            }
            self.inflight.insert(p);
            if self.synchronous() {
                let storage = self.gen.generate_chunk(p);
                let _ = self.result_tx.send((p, storage));
            } else {
                self.queue.push(p);
            }
        }
    }

    /// Pull finished chunks out of the result channel and add them to the world.
    fn integrate_results(&mut self, center: ChunkPos) {
        let mut uploaded = 0;
        while uploaded < self.config.max_uploads_per_update {
            match self.result_rx.try_recv() {
                Ok((pos, storage)) => {
                    self.inflight.remove(&pos);
                    // It may have drifted out of range while generating.
                    if !self.in_load_range(pos, center) {
                        continue;
                    }
                    self.world.insert_chunk(pos, storage);
                    // The new chunk and its neighbours need (re)meshing so the
                    // shared borders are culled correctly.
                    self.mark_dirty_with_neighbours(pos);
                    uploaded += 1;
                }
                Err(_) => break,
            }
        }
    }

    fn mark_dirty_with_neighbours(&mut self, pos: ChunkPos) {
        for (dx, dy, dz) in [
            (0, 0, 0),
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            if let Some(c) = self.world.get_chunk_mut(pos.offset(dx, dy, dz)) {
                c.dirty = true;
            }
        }
    }

    fn in_load_range(&self, pos: ChunkPos, center: ChunkPos) -> bool {
        let vd = self.config.view_distance;
        let dx = pos.x - center.x;
        let dz = pos.z - center.z;
        dx * dx + dz * dz <= vd * vd
            && pos.y >= self.config.min_chunk_y
            && pos.y <= self.config.max_chunk_y
    }

    /// Remove chunks beyond the view distance plus hysteresis margin.
    fn unload_distant(&mut self, center: ChunkPos) {
        let limit = self.config.view_distance + self.config.unload_margin;
        let limit_sq = limit * limit;
        let to_remove: Vec<ChunkPos> = self
            .world
            .iter_chunks()
            .filter_map(|c| {
                let dx = c.pos.x - center.x;
                let dz = c.pos.z - center.z;
                if dx * dx + dz * dz > limit_sq {
                    Some(c.pos)
                } else {
                    None
                }
            })
            .collect();
        for pos in to_remove {
            // Preserve player edits before dropping the chunk from memory.
            if let Some(chunk) = self.world.remove_chunk(pos) {
                if chunk.modified {
                    self.edits.insert(pos, chunk.storage);
                }
            }
            self.meshes.remove(&pos);
            self.mesh_versions.remove(&pos);
        }
    }

    /// (Re)mesh dirty chunks, nearest first, within the per-update budget.
    ///
    /// Meshing is the heaviest per-frame CPU cost, so when worker threads are
    /// available the budgeted batch is meshed in parallel. Each chunk only reads
    /// the (immutable, `Sync`) world, so a scoped thread pool can fan the batch
    /// out with shared `&World` — no copying or locking of voxel data.
    fn remesh_dirty(&mut self, center: ChunkPos) {
        let mut dirty: Vec<ChunkPos> = self
            .world
            .iter_chunks()
            .filter(|c| c.dirty)
            .map(|c| c.pos)
            .collect();
        dirty.sort_by_key(|p| p.distance_sq(center));
        dirty.truncate(self.config.max_meshes_per_update);
        if dirty.is_empty() {
            return;
        }

        let world = &self.world;
        let registry: &BlockRegistry = &self.registry;
        let mesh_one = |pos: ChunkPos| -> Option<(ChunkPos, ChunkMesh)> {
            let sampler = ChunkNeighborSampler::new(world, pos)?;
            Some((pos, mesh_chunk(&sampler, registry)))
        };

        let results: Vec<(ChunkPos, ChunkMesh)> = if self.mesh_threads <= 1 || dirty.len() == 1 {
            dirty.iter().filter_map(|&p| mesh_one(p)).collect()
        } else {
            let threads = self.mesh_threads.min(dirty.len());
            let per = dirty.len().div_ceil(threads);
            std::thread::scope(|scope| {
                let handles: Vec<_> = dirty
                    .chunks(per)
                    .map(|slice| scope.spawn(|| slice.iter().filter_map(|&p| mesh_one(p)).collect::<Vec<_>>()))
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap())
                    .collect()
            })
        };

        for (pos, mesh) in results {
            if mesh.is_empty() {
                self.meshes.remove(&pos);
            } else {
                self.meshes.insert(pos, mesh);
            }
            // Bump the version even when the mesh became empty, so the renderer
            // drops a now-cleared chunk's stale buffers.
            *self.mesh_versions.entry(pos).or_insert(0) += 1;
            if let Some(c) = self.world.get_chunk_mut(pos) {
                c.dirty = false;
            }
        }
    }

    /// Edit a single block and mark the affected chunk(s) for remeshing. The
    /// change takes visual effect on a subsequent `update`.
    pub fn set_block(&mut self, pos: BlockPos, block: BlockId) -> bool {
        self.world.set_block(pos, block)
    }

    /// True if there are no outstanding generation jobs or dirty chunks — i.e.
    /// the world around the player is fully streamed in and meshed.
    pub fn is_settled(&self) -> bool {
        self.inflight.is_empty() && self.world.iter_chunks().all(|c| !c.dirty)
    }

    pub fn meshes(&self) -> impl Iterator<Item = (&ChunkPos, &ChunkMesh)> {
        self.meshes.iter()
    }

    /// Current mesh revision for a chunk; changes whenever it is remeshed.
    pub fn mesh_version(&self, pos: ChunkPos) -> u64 {
        self.mesh_versions.get(&pos).copied().unwrap_or(0)
    }

    /// All player edits, including chunks still loaded — the complete set needed
    /// to persist the world. Merges the unload cache with live modified chunks.
    pub fn collect_edits(&self) -> Vec<(ChunkPos, ChunkStorage)> {
        let mut map: AHashMap<ChunkPos, ChunkStorage> = self.edits.clone();
        for chunk in self.world.iter_chunks() {
            if chunk.modified {
                map.insert(chunk.pos, chunk.storage.clone());
            }
        }
        map.into_iter().collect()
    }

    /// Inject saved edits (on load). They take effect as chunks stream in.
    pub fn restore_edits(&mut self, edits: impl IntoIterator<Item = (ChunkPos, ChunkStorage)>) {
        for (pos, storage) in edits {
            self.edits.insert(pos, storage);
        }
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    pub fn stats(&self) -> StreamStats {
        StreamStats {
            loaded_chunks: self.world.chunk_count(),
            meshed_chunks: self.meshes.len(),
            inflight: self.inflight.len(),
            voxel_memory_bytes: self.world.voxel_memory_bytes(),
        }
    }
}

impl Drop for ChunkManager {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.queue.shutdown();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn small_config(vd: i32) -> StreamConfig {
        StreamConfig {
            view_distance: vd,
            min_chunk_y: 0,
            max_chunk_y: 3,
            unload_margin: 1,
            // Large budgets so a few updates fully settle in tests.
            max_uploads_per_update: 4096,
            max_meshes_per_update: 4096,
            max_jobs_per_update: 4096,
        }
    }

    /// Run updates until the world settles or a step cap is hit. A short sleep
    /// between steps gives background generation workers time to produce results
    /// (a no-op for the synchronous path, which settles in a couple of steps).
    fn settle(mgr: &mut ChunkManager, center: Vec3, max_steps: usize) {
        for _ in 0..max_steps {
            mgr.update(center);
            if mgr.is_settled() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    fn expected_columns(vd: i32) -> usize {
        let mut n = 0;
        for dz in -vd..=vd {
            for dx in -vd..=vd {
                if dx * dx + dz * dz <= vd * vd {
                    n += 1;
                }
            }
        }
        n
    }

    #[test]
    fn synchronous_streaming_loads_expected_region() {
        let mut mgr = ChunkManager::new(42, small_config(2), 0);
        settle(&mut mgr, Vec3::new(8.0, 80.0, 8.0), 10);
        let cols = expected_columns(2);
        let layers = (3 - 0 + 1) as usize;
        assert_eq!(mgr.world.chunk_count(), cols * layers);
        assert!(mgr.is_settled());
        // At least some chunks should have produced geometry near the surface.
        assert!(mgr.mesh_count() > 0);
    }

    #[test]
    fn moving_unloads_and_loads() {
        let mut mgr = ChunkManager::new(7, small_config(2), 0);
        settle(&mut mgr, Vec3::new(0.0, 80.0, 0.0), 10);
        let initial = mgr.world.chunk_count();
        assert!(initial > 0);

        // Walk far along +X so the original region falls out of range.
        settle(&mut mgr, Vec3::new(2000.0, 80.0, 0.0), 10);
        assert!(mgr.is_settled());
        // The far origin chunk must have been unloaded.
        assert!(!mgr.world.contains_chunk(ChunkPos::new(0, 1, 0)));
        // Region size stays bounded.
        assert_eq!(mgr.world.chunk_count(), initial);
    }

    #[test]
    fn threaded_workers_produce_same_world_as_sync() {
        // Generation must be identical regardless of threading.
        let mut sync_mgr = ChunkManager::new(123, small_config(2), 0);
        settle(&mut sync_mgr, Vec3::new(8.0, 80.0, 8.0), 20);

        let mut thr_mgr = ChunkManager::new(123, small_config(2), 3);
        settle(&mut thr_mgr, Vec3::new(8.0, 80.0, 8.0), 200);

        assert_eq!(sync_mgr.world.chunk_count(), thr_mgr.world.chunk_count());
        // Spot-check a column of blocks matches between the two.
        for y in 60..96 {
            let p = BlockPos::new(5, y, 5);
            assert_eq!(
                sync_mgr.world.block_at(p),
                thr_mgr.world.block_at(p),
                "block mismatch at {p:?}"
            );
        }
    }

    #[test]
    fn budget_limits_work_per_update() {
        let mut cfg = small_config(3);
        cfg.max_jobs_per_update = 5;
        cfg.max_uploads_per_update = 5;
        cfg.max_meshes_per_update = 2;
        let mut mgr = ChunkManager::new(1, cfg, 0);
        // One update should not load the entire region at once.
        mgr.update(Vec3::new(0.0, 80.0, 0.0));
        assert!(mgr.world.chunk_count() <= 5);
        assert!(!mgr.is_settled());
        // But it does settle eventually.
        settle(&mut mgr, Vec3::new(0.0, 80.0, 0.0), 200);
        assert!(mgr.is_settled());
    }

    #[test]
    fn edits_persist_across_unload_and_reload() {
        let mut mgr = ChunkManager::new(99, small_config(1), 0);
        settle(&mut mgr, Vec3::new(8.0, 80.0, 8.0), 10);
        let p = BlockPos::new(8, 80, 8);
        let original = mgr.world.block_at(p);
        mgr.set_block(p, pixelcraft_core::block::blocks::LANTERN);
        assert_eq!(mgr.world.block_at(p), pixelcraft_core::block::blocks::LANTERN);

        // Wander far so the edited chunk unloads, then come back.
        settle(&mut mgr, Vec3::new(5000.0, 80.0, 5000.0), 10);
        assert!(!mgr.world.contains_chunk(p.chunk()));
        settle(&mut mgr, Vec3::new(8.0, 80.0, 8.0), 10);

        // The edit must have survived the round-trip.
        assert_eq!(mgr.world.block_at(p), pixelcraft_core::block::blocks::LANTERN);
        assert_ne!(original, pixelcraft_core::block::blocks::LANTERN);
    }

    #[test]
    fn block_edits_take_effect() {
        let mut mgr = ChunkManager::new(99, small_config(1), 0);
        settle(&mut mgr, Vec3::new(8.0, 80.0, 8.0), 10);
        let p = BlockPos::new(8, 80, 8);
        assert!(mgr.set_block(p, pixelcraft_core::block::blocks::LANTERN));
        assert_eq!(mgr.world.block_at(p), pixelcraft_core::block::blocks::LANTERN);
    }
}
