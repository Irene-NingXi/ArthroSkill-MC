# Clip data contract

Store trusted local `.pt` clips under `processed/{train,val,test}/clips/`.

| Field | Meaning |
| --- | --- |
| frames | Float tensor [3, 16, 480, 640], per-video normalised RGB |
| motion | Float tensor [16, 6], image-coordinate kinematic descriptors |
| detection_conf | Float tensor [16], highest instrument confidence per frame; zero when absent |
| label_cls | Integer 0/1/2 for novice/intermediate/expert |
| label_reg | Float tensor [7], six consensus 1–5 dimensions followed by consensus 6–30 total |
| score_units | Literal `grs_points` |
| video_id | Procedure identifier, shared by all its clips |
| centre | Site A, B or C |
| source_frame_indices | Original indices of the retained frames |

Missing motion or confidence raises an error instead of inventing valid observations. Legacy normalised labels must be regenerated. Keep every procedure's clips in a single partition. For LOSO, use independently prepared site-held-out folds as described in [the protocol](../docs/protocol.md).

The workbook provides ratings, predictions and assignments; it contains no raw videos. `prepare_metadata.py` exports labels and authoritative assignments locally without publishing source records.

## SAIS feature cache

SAIS uses a separate paired frozen-DINO cache with an authoritative `manifest.json`, under `sais_features/{train,val,test}/clips`. It follows the same procedure assignment and 16-frame source indices. RGB and RAFT-flow embeddings have dimension 384. Each payload records flow endpoint pairs; the manifest records source and weight checksums. Instrument kinematics are not a substitute for optical flow. See [SAIS integration](../docs/sais.md) for extraction and validation.
