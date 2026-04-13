# Train Pretraining Bugs

Review of [src/training/train_pretraining.py](../src/training/train_pretraining.py).

## `with wandb.init(...)` structure — OK

- Context manager form used correctly.
- `run.name` and `run.id` accessed inside the block.
- Training loop lives inside the `with` block, so `wandb.finish()` runs on normal exit and on exception.

## Bugs to fix before first run

### 1. Path concatenation will raise `TypeError`

[train_pretraining.py:31](../src/training/train_pretraining.py#L31)

```python
os.makedirs(config.checkpoints_dir / Path(run_tag), exist_ok=True)
```

`config.checkpoints_dir` is a `str` ([pcd_config.py:63](../src/pcd_config.py#L63)), and `str / Path` raises `TypeError`.

Fix:
```python
run_checkpoint_dir = Path(config.checkpoints_dir) / run_tag
os.makedirs(run_checkpoint_dir, exist_ok=True)
```

Also: the per-run dir is created but never handed to `save_checkpoint` at lines 165 and 168. Unless `save_checkpoint` derives the path from `config` + `run.name` internally, checkpoints land in the top-level `checkpoints/` folder and the per-run subfolder stays empty.

### 2. `end_extract` produces an empty slice

[train_pretraining.py:111](../src/training/train_pretraining.py#L111)

```python
activations = subject.get_middle_activations(
    tokens=subject_inputs,
    attention_mask=torch.ones_like(subject_inputs),
    start_extract=config.n_prefix,   # 16
    end_extract=config.n_middle,      # 16  ← index, not length
)
```

`get_middle_activations` slices `resid[:, start_extract:end_extract, :]` ([subject_model.py:63](../src/architecture/subject_model.py#L63)). With `start=16, end=16`, the slice is **empty**, so `sparse_embedding` has shape `[batch, 0, d_model]` and downstream breaks silently.

Fix:
```python
end_extract=config.n_prefix + config.n_middle   # 32
```

### 3. Scheduler / `max_train_steps` mis-accounting

`global_step` is incremented every micro-batch ([train_pretraining.py:145](../src/training/train_pretraining.py#L145)), but `scheduler.step()` only runs on optimizer updates ([train_pretraining.py:142](../src/training/train_pretraining.py#L142)). With `grad_accum_steps=4` and `max_train_steps=4000`, the scheduler is configured with `T_max=4000` but only advances 1000 times — LR decays to `cos(π/4) ≈ 0.707 × base_lr` and never reaches the trough.

Pick one fix:

- **Compensate `T_max`:**
  ```python
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=config.max_train_steps // config.grad_accum_steps
  )
  ```

- **Count optimizer steps instead (preferred):** keep a separate `opt_step` counter that increments only when you call `optimizer.step()`. Loop while `opt_step < max_train_steps`. Then `max_train_steps` unambiguously means "optimizer updates," which is how papers quote it.

## Polish (non-blocking)

### 4. `config.use_wandb` is ignored

The `if config.use_wandb:` guard was removed, so wandb is always on. To preserve the toggle, use wandb's built-in disabled mode:

```python
with wandb.init(
    project=config.wandb_project,
    name=wandb_run_name,
    mode="online" if config.use_wandb else "disabled",
    config={...},
) as run:
```

In disabled mode, `wandb.log(...)` is a no-op, so the loop body stays unchanged.

### 5. `autocast(device_type=config.device)` — brittle to device strings

[train_pretraining.py:116](../src/training/train_pretraining.py#L116)

Works today because `config.device` is `"cuda"` or `"cpu"`. Breaks if you ever set `"cuda:0"` — autocast wants just the type, not the index.

Safer:
```python
device_type = "cuda" if torch.cuda.is_available() else "cpu"
with torch.autocast(device_type=device_type, dtype=config.dtype):
```

### 6. Inconsistent parameter access between optimizer and clipping

- Optimizer ([train_pretraining.py:74](../src/training/train_pretraining.py#L74)): `decoder.parameters()`
- Clipping ([train_pretraining.py:137](../src/training/train_pretraining.py#L137)): `decoder.model.parameters()`

Yields the same tensors in practice (since `DecoderModel.self.model` is the only submodule with params), but worth making them match to avoid future confusion.
