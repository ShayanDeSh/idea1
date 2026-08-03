use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{EngineError, EngineResult};

/// Resolved safetensors weight files for a model directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightFiles {
    /// Ordered list of `.safetensors` shard paths (or a single file).
    pub paths: Vec<PathBuf>,
    /// Optional tensor-name → shard filename map from an index file.
    pub weight_map: Option<HashMap<String, String>>,
}

impl WeightFiles {
    pub fn is_sharded(&self) -> bool {
        self.paths.len() > 1 || self.weight_map.is_some()
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Discover weight files under `model_dir`.
///
/// Resolution order:
/// 1. `model.safetensors.index.json` (sharded) → unique shard filenames from `weight_map`
/// 2. `model.safetensors` (single file)
/// 3. `model-00001-of-NNNNN.safetensors` style shards (sorted), if present
pub fn discover_weight_files(model_dir: impl AsRef<Path>) -> EngineResult<WeightFiles> {
    let model_dir = model_dir.as_ref();
    if !model_dir.is_dir() {
        return Err(EngineError::InvalidModelDir {
            path: model_dir.to_path_buf(),
            reason: "path is not a directory".into(),
        });
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        return discover_from_index(model_dir, &index_path);
    }

    let single = model_dir.join("model.safetensors");
    if single.is_file() {
        return Ok(WeightFiles {
            paths: vec![single],
            weight_map: None,
        });
    }

    let mut shards = discover_numbered_shards(model_dir)?;
    if !shards.is_empty() {
        shards.sort();
        return Ok(WeightFiles {
            paths: shards,
            weight_map: None,
        });
    }

    Err(EngineError::weights(format!(
        "no safetensors weights found in {}",
        model_dir.display()
    )))
}

fn discover_from_index(model_dir: &Path, index_path: &Path) -> EngineResult<WeightFiles> {
    let file = File::open(index_path)?;
    let reader = BufReader::new(file);
    let index: SafetensorsIndex = serde_json::from_reader(reader)?;

    if index.weight_map.is_empty() {
        return Err(EngineError::weights(format!(
            "empty weight_map in {}",
            index_path.display()
        )));
    }

    let mut unique_names = BTreeSet::new();
    for name in index.weight_map.values() {
        unique_names.insert(name.clone());
    }

    let mut paths = Vec::with_capacity(unique_names.len());
    for name in unique_names {
        let path = model_dir.join(&name);
        if !path.is_file() {
            return Err(EngineError::MissingFile(path));
        }
        paths.push(path);
    }

    Ok(WeightFiles {
        paths,
        weight_map: Some(index.weight_map),
    })
}

fn discover_numbered_shards(model_dir: &Path) -> EngineResult<Vec<PathBuf>> {
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if is_numbered_shard_name(name) {
            shards.push(path);
        }
    }
    Ok(shards)
}

fn is_numbered_shard_name(name: &str) -> bool {
    // e.g. model-00001-of-00002.safetensors
    let Some(stem) = name.strip_suffix(".safetensors") else {
        return false;
    };
    let Some(rest) = stem.strip_prefix("model-") else {
        return false;
    };
    let parts: Vec<_> = rest.split("-of-").collect();
    parts.len() == 2
        && parts[0].len() == 5
        && parts[1].len() == 5
        && parts[0].chars().all(|c| c.is_ascii_digit())
        && parts[1].chars().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn discovers_single_model_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, b"fake").unwrap();

        let files = discover_weight_files(dir.path()).unwrap();
        assert_eq!(files.paths, vec![path]);
        assert!(!files.is_sharded());
    }

    #[test]
    fn discovers_sharded_via_index() {
        let dir = tempfile::tempdir().unwrap();
        let shard_a = dir.path().join("model-00001-of-00002.safetensors");
        let shard_b = dir.path().join("model-00002-of-00002.safetensors");
        std::fs::write(&shard_a, b"a").unwrap();
        std::fs::write(&shard_b, b"b").unwrap();

        let index = serde_json::json!({
            "metadata": {"total_size": 2},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        });
        let mut index_file =
            File::create(dir.path().join("model.safetensors.index.json")).unwrap();
        write!(index_file, "{index}").unwrap();

        let files = discover_weight_files(dir.path()).unwrap();
        assert!(files.is_sharded());
        assert_eq!(files.paths, vec![shard_a, shard_b]);
        assert_eq!(
            files.weight_map.as_ref().unwrap().get("lm_head.weight"),
            Some(&"model-00002-of-00002.safetensors".to_string())
        );
    }

    #[test]
    fn discovers_numbered_shards_without_index() {
        let dir = tempfile::tempdir().unwrap();
        let shard_b = dir.path().join("model-00002-of-00002.safetensors");
        let shard_a = dir.path().join("model-00001-of-00002.safetensors");
        std::fs::write(&shard_b, b"b").unwrap();
        std::fs::write(&shard_a, b"a").unwrap();

        let files = discover_weight_files(dir.path()).unwrap();
        assert_eq!(files.paths, vec![shard_a, shard_b]);
    }

    #[test]
    fn errors_when_no_weights() {
        let dir = tempfile::tempdir().unwrap();
        let err = discover_weight_files(dir.path()).unwrap_err();
        assert!(matches!(err, EngineError::Weights(_)));
    }
}
