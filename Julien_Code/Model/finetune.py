from collections import defaultdict
from typing import Callable

from uni2ts.model.moirai import MoiraiFinetune
from uni2ts.model.moirai.module import MoiraiModule
from uni2ts.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    DummyValueImputation,
    ExtendMask,
    FixedPatchSizeConstraints,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SelectFields,
    SequencifyField,
    Transformation,
)


class Finetuner(MoiraiFinetune):
    """Thin wrapper around `MoiraiPretrain` that keeps a freq attribute but
    **does not** forward it to the parent constructor (which doesn't accept it).
    Everything else (optimiser, transforms, trainer factory) is inherited.
    """

    def __init__(
        self,
        module: MoiraiModule,
        min_patches = 2,
        min_mask_ratio = 0.2,
        max_mask_ratio = 0.4,
        max_dim = 128, # 256
        num_training_steps = 20000,
        num_warmup_steps = 1_000,
        learning_rate = 1e-3,
        weight_decay = 1e-2,
    ) -> None:
        super().__init__(
            min_patches = min_patches,
            min_mask_ratio = min_mask_ratio,
            max_mask_ratio = max_mask_ratio,
            max_dim = max_dim, # 256
            num_training_steps = num_training_steps,
            num_warmup_steps = num_warmup_steps,
            module_kwargs = None,
            module = module,
            num_samples = 100,
            lr = learning_rate,
            weight_decay = weight_decay
        )

    @property
    def train_transform_map(
        self,
    ) -> dict[str | type, Callable[..., Transformation]]:
        def default_train_transform(patch_size: int):
            return (
                GetPatchSize(
                    min_time_patches=self.hparams["min_patches"],
                    target_field="target",
                    patch_sizes=self.module.patch_sizes,
                    patch_size_constraints=FixedPatchSizeConstraints(
                        start=patch_size
                    ),
                    offset=True,
                )
                + PatchCrop(
                    min_time_patches=self.hparams["min_patches"],
                    max_patches=self.module.max_seq_len,
                    will_flatten=True,
                    offset=True,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                )
                + PackFields(
                    output_field="past_feat_dynamic_real",
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=max(self.module.patch_sizes),
                    fields=("target", "observed_mask"),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams["max_dim"],
                    randomize=True,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + MaskedPrediction(
                    min_mask_ratio=self.hparams["min_mask_ratio"],
                    max_mask_ratio=self.hparams["max_mask_ratio"],
                    target_field="target",
                    truncate_fields=("variate_id", "time_id", "observed_mask"),
                    optional_truncate_fields=("past_feat_dynamic_real",),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + ExtendMask(
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )

        return defaultdict(lambda: default_train_transform)

# Usage example (unchanged):
# finetune_module = MoiraiFinetune(
#     module_kwargs={"patch_size":8, "context_length":256, "prediction_length":20},
#     freq="S", device="cuda")
# collate_fn = finetune_module.train_transform_map[type(ds)]()
# trainer    = finetune_module.create_trainer(max_epochs=10)
# trainer.fit(finetune_module, loader)
