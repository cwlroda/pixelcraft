//! A gentle quest chain that gives the cat cosy goals: gather flowers, pick
//! berries, chop wood, build a little home, light it with lanterns. Each quest
//! tracks progress toward a target and hands out a small reward on completion,
//! then the next begins. Pure state machine — no rendering — so it's testable.

use pixelcraft_core::block::{blocks, BlockId};

/// What a quest asks the player to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Objective {
    /// Collect (mine/harvest) any of these block types.
    Collect(Vec<BlockId>),
    /// Place any of these block types.
    Place(Vec<BlockId>),
}

#[derive(Clone, Debug)]
pub struct Quest {
    pub title: &'static str,
    pub objective: Objective,
    pub target: u32,
    /// Items granted to the player on completion.
    pub reward: Vec<(BlockId, u32)>,
}

impl Quest {
    fn matches_collect(&self, id: BlockId) -> bool {
        matches!(&self.objective, Objective::Collect(set) if set.contains(&id))
    }
    fn matches_place(&self, id: BlockId) -> bool {
        matches!(&self.objective, Objective::Place(set) if set.contains(&id))
    }
}

/// Tracks the active quest and progress through the chain.
pub struct QuestLog {
    quests: Vec<Quest>,
    current: usize,
    progress: u32,
}

impl QuestLog {
    pub fn new(quests: Vec<Quest>) -> Self {
        Self {
            quests,
            current: 0,
            progress: 0,
        }
    }

    /// The cosy starter chain.
    pub fn cosy_chain() -> Self {
        Self::new(vec![
            Quest {
                title: "GATHER WILDFLOWERS",
                objective: Objective::Collect(vec![blocks::FLOWER_PINK, blocks::FLOWER_BLUE]),
                target: 5,
                reward: vec![(blocks::PLANK, 16)],
            },
            Quest {
                title: "PICK SWEET BERRIES",
                objective: Objective::Collect(vec![blocks::BERRY_BUSH]),
                target: 6,
                reward: vec![(blocks::LANTERN, 2)],
            },
            Quest {
                title: "CHOP COSY TIMBER",
                objective: Objective::Collect(vec![blocks::TRUNK]),
                target: 8,
                reward: vec![(blocks::PLANK, 24)],
            },
            Quest {
                title: "BUILD A LITTLE HOME",
                objective: Objective::Place(vec![blocks::PLANK]),
                target: 20,
                reward: vec![(blocks::GLASS, 8), (blocks::LANTERN, 2)],
            },
            Quest {
                title: "LIGHT UP THE NIGHT",
                objective: Objective::Place(vec![blocks::LANTERN]),
                target: 4,
                reward: vec![(blocks::CRYSTAL, 4)],
            },
        ])
    }

    /// The quest currently in progress, if any remain.
    pub fn current(&self) -> Option<&Quest> {
        self.quests.get(self.current)
    }

    pub fn is_complete(&self) -> bool {
        self.current >= self.quests.len()
    }

    pub fn completed_count(&self) -> usize {
        self.current
    }

    /// Objective text + (done, total) for the HUD.
    pub fn hud(&self) -> Option<(String, (u32, u32))> {
        self.current().map(|q| {
            (q.title.to_string(), (self.progress.min(q.target), q.target))
        })
    }

    /// Report a collected block. Returns any reward to grant if this completed
    /// the active quest.
    pub fn on_collect(&mut self, id: BlockId, count: u32) -> Vec<(BlockId, u32)> {
        if self.current().map(|q| q.matches_collect(id)).unwrap_or(false) {
            self.advance_progress(count)
        } else {
            Vec::new()
        }
    }

    /// Report a placed block. Returns any completion reward.
    pub fn on_place(&mut self, id: BlockId) -> Vec<(BlockId, u32)> {
        if self.current().map(|q| q.matches_place(id)).unwrap_or(false) {
            self.advance_progress(1)
        } else {
            Vec::new()
        }
    }

    fn advance_progress(&mut self, by: u32) -> Vec<(BlockId, u32)> {
        let Some(quest) = self.quests.get(self.current) else {
            return Vec::new();
        };
        self.progress += by;
        if self.progress >= quest.target {
            let reward = quest.reward.clone();
            self.current += 1;
            self.progress = 0;
            reward
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_advances_and_completes_with_reward() {
        let mut log = QuestLog::cosy_chain();
        // First quest: gather 5 flowers.
        assert_eq!(log.hud().unwrap().1, (0, 5));
        let r = log.on_collect(blocks::FLOWER_PINK, 2);
        assert!(r.is_empty());
        assert_eq!(log.hud().unwrap().1, (2, 5));
        // Wrong block type doesn't count.
        let r = log.on_collect(blocks::STONE, 10);
        assert!(r.is_empty());
        assert_eq!(log.hud().unwrap().1, (2, 5));
        // Finish it; reward is planks and we advance to quest 2.
        let r = log.on_collect(blocks::FLOWER_BLUE, 3);
        assert_eq!(r, vec![(blocks::PLANK, 16)]);
        assert_eq!(log.completed_count(), 1);
        assert_eq!(log.current().unwrap().title, "PICK SWEET BERRIES");
    }

    #[test]
    fn place_quest_tracks_placement() {
        let mut log = QuestLog::new(vec![Quest {
            title: "BUILD",
            objective: Objective::Place(vec![blocks::PLANK]),
            target: 3,
            reward: vec![(blocks::LANTERN, 1)],
        }]);
        assert!(log.on_place(blocks::PLANK).is_empty());
        assert!(log.on_place(blocks::DIRT).is_empty()); // wrong type
        assert!(log.on_place(blocks::PLANK).is_empty());
        let r = log.on_place(blocks::PLANK);
        assert_eq!(r, vec![(blocks::LANTERN, 1)]);
        assert!(log.is_complete());
        assert!(log.hud().is_none());
    }

    #[test]
    fn completing_chain_ends_cleanly() {
        let mut log = QuestLog::new(vec![Quest {
            title: "ONE",
            objective: Objective::Collect(vec![blocks::TRUNK]),
            target: 1,
            reward: vec![],
        }]);
        log.on_collect(blocks::TRUNK, 1);
        assert!(log.is_complete());
        // Further reports are harmless no-ops.
        assert!(log.on_collect(blocks::TRUNK, 5).is_empty());
    }
}
