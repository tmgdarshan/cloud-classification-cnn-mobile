# config/evaluation/

Evaluation protocols answer one question: **how is performance measured?**

## Profiles

- `standard.toml` — primary + secondary metrics, confusion matrix, report.

## Keys

| Key | Meaning |
|-----|---------|
| `name` | Profile name (required). |
| `approval_status` | `draft` / `validation` / `approved`. |
| `primary_metric` | The single headline metric. |
| `secondary_metrics` | Additional reported metrics. |
| `produce_confusion_matrix` | Whether to emit a confusion matrix. |
| `produce_classification_report` | Whether to emit a per-class report. |

Keeping evaluation separate means the same measurement methodology can be reused
across experiments and models.
