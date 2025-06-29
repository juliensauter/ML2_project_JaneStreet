import torch

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

class Model(MoiraiForecast):
    
    def __init__(
        self,
        path,
        prediction_length,
        context_length,
        patch_size,
        num_samples,
        target_dim,
        feat_dynamic_real_dim,
        past_feat_dynamic_real_dim,
        from_disk,
        device,
        base_model_size: str = "small"
        ):
        """_summary_
        Args:
            path (string): either path on disk (if from_disk=True), or size of pretrained moirai model (from web)
            from_disk (bool, optional): _description_. Defaults to False.
            device (string): _description_
            base_model_size (str, optional): The base model size to use when loading from disk. Defaults to "small".
        """
        # When loading from disk, we assume `path` is the file path to the weights,
        # and a base model still needs to be loaded from the web.
        # This implementation requires a hardcoded base model size when from_disk is True.
        # A better implementation would pass the base model size as an argument.
        super().__init__(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{path if not from_disk else base_model_size}"),
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        )
        
        if from_disk:
            self.load_model(path, device)

        self.to(device)
    
    def load_model(self, path, device=None):
        if device is None:
            # self.device is a property of LightningModule
            device = self.device
        
        # We are loading the state_dict of the inner MoiraiModule
        self.module.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        self.to(device)
    
    def save_model(self, path):
        # We are saving the state_dict of the inner MoiraiModule
        torch.save(self.module.state_dict(), path)
    
    def inference(self, dataloader, device):
        pass