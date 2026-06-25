import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .preprocessor import StepPreprocessor


class AgentTrajectoryDataset(Dataset):
    """
    Each item is one complete agent trajectory.
    The DataLoader must use batch_size=1 because trajectories have variable
    numbers of steps — there is no cross-trajectory batching.
    """

    def __init__(self, json_path, preprocessor=None, max_len=512, max_steps=20):
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.preprocessor = preprocessor or StepPreprocessor(max_len=max_len)
        self.max_steps = max_steps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        is_hal = sample.get("is_hallucination", False)
        if isinstance(is_hal, str):
            is_hal = is_hal.lower() == "true"

        hal_step = sample.get("hallucination_step")
        if hal_step is not None:
            hal_step = int(hal_step)

        steps = self.preprocessor.encode_trajectory(sample)

        # Graceful empty-trajectory handling
        if not steps:
            ml = self.preprocessor.max_len
            return {
                "input_ids": torch.zeros((1, ml), dtype=torch.long),
                "attention_mask": torch.zeros((1, ml), dtype=torch.long),
                "hal_label": torch.tensor([0.0]),
            }

        # Truncation: if the trajectory is longer than max_steps, keep a window
        # that always INCLUDES the annotated hallucination step (if known).
        n = len(steps)
        if n > self.max_steps:
            if is_hal and hal_step is not None:
                # Find the position of the hallucination step
                hal_pos = next(
                    (i for i, s in enumerate(steps) if s["step_idx"] == hal_step),
                    n - 1,
                )
                # Build a window of max_steps centred on hal_pos
                half = self.max_steps // 2
                start = max(0, hal_pos - half)
                end = min(n, start + self.max_steps)
                start = max(0, end - self.max_steps)
                steps = steps[start:end]
            else:
                # No annotation — keep the last max_steps (model likely hallucinates late)
                steps = steps[-self.max_steps:]

        input_ids = torch.stack(
            [s["encoding"]["input_ids"].squeeze(0) for s in steps]
        )
        attention_masks = torch.stack(
            [s["encoding"]["attention_mask"].squeeze(0) for s in steps]
        )
        hal_labels = torch.tensor([float(s["label"]) for s in steps])

        return {
            "input_ids": input_ids,           # (N_steps, max_len)
            "attention_mask": attention_masks, # (N_steps, max_len)
            "hal_label": hal_labels,           # (N_steps,)
        }

    def get_class_weights(self):
        """
        Returns per-sample weights for WeightedRandomSampler so that each
        training batch has roughly equal representation of hallucinated and
        clean trajectories.
        """
        is_hal_flags = []
        for s in self.samples:
            flag = s.get("is_hallucination", False)
            if isinstance(flag, str):
                flag = flag.lower() == "true"
            is_hal_flags.append(int(flag))

        n_hal = sum(is_hal_flags)
        n_clean = len(is_hal_flags) - n_hal
        w_hal = 1.0 / n_hal if n_hal else 1.0
        w_clean = 1.0 / n_clean if n_clean else 1.0

        weights = [w_hal if f else w_clean for f in is_hal_flags]
        return weights


def get_dataloader(json_path, preprocessor, batch_size=1, shuffle=True,
                   use_weighted_sampler=False):
    """
    batch_size must be 1 because each item is a variable-length trajectory.
    use_weighted_sampler=True balances hallucinated vs. clean at the batch level.
    """
    dataset = AgentTrajectoryDataset(json_path, preprocessor=preprocessor)

    if use_weighted_sampler and shuffle:
        weights = dataset.get_class_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
