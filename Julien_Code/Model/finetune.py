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
    """
    Wrapper for MoiraiFinetune that implements layer freezing for stable fine-tuning.
    Only the input embedding and output head layers are trained.
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
        self._freeze_layers()

    def _freeze_layers(self, bitfit: bool = True) -> None:
        """
        Freeze embedding + lower 18 encoder blocks; 
        leave top-6 blocks, final norm, and param head trainable.
        Optionally BitFit-unfreeze all biases & norm-scales.
        """
        # 1) freeze everything
        for p in self.module.parameters():
            p.requires_grad = False

        # 2) unfreeze *top-6* encoder blocks
        # The Moirai "large" model has 24 total encoder layers. We train the top 6.
        num_layers = len(self.module.encoder.layers)
        num_unfreeze = 6
        for block in self.module.encoder.layers[num_layers - num_unfreeze:]:
            for p in block.parameters():
                p.requires_grad = True

        # 3) unfreeze the final RMSNorm after the stack
        for p in self.module.encoder.norm.parameters():
            p.requires_grad = True

        # 4) unfreeze the distribution head, which is named 'param_proj'
        for p in self.module.param_proj.parameters():
            p.requires_grad = True

        # 5) (optional) BitFit – also thaw every bias & norm-scale everywhere
        if bitfit:
            for name, p in self.module.named_parameters():
                if name.endswith(".bias") or ".norm.weight" in name:
                    p.requires_grad = True

        # 6) bookkeeping
        trainable = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.module.parameters())
        print(f"❄️  Frozen {total-trainable:,}/{total:,} params  ·  Trainable = {trainable/1e6:.1f} M  ({100*trainable/total:.2f} %)")

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