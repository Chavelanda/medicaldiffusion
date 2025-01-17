import os
from einops import rearrange
from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F

from ddpm.utils import video_tensor_to_gif


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

class SampleAndSaveCallback(Callback):

    def __init__(self, results_folder, sample_every_n_epochs=10, save_gif=True, save_image=False, save_func=None):
        super().__init__()
        assert save_gif or save_image, "At least one of save_gif or save_image must be True"
        assert not save_image or save_func is not None, "save_func must be provided if save_image is True"

        self.results_folder = results_folder
        self.sample_every_n_epochs = sample_every_n_epochs
        
        self.save_gif = save_gif
        self.save_image = save_image
        self.save_func = save_func

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.sample_every_n_epochs != 0 or pl_module.global_rank != 0:
            return
        
        image = pl_module.sample_with_random_cond()
        name = f'epoch_{trainer.current_epoch}_step_{trainer.global_step}'

        if self.save_image:
            self.save_func(name, image, self.results_folder)

        if self.save_gif:
            image = F.pad(image, (2, 2, 2, 2))
            gif = rearrange(
                image, '(i j) c f h w -> c f (i h) (j w)', i=1)
            video_path = os.path.join(self.results_folder, f'{name}.gif')
            video_tensor_to_gif(gif, video_path)