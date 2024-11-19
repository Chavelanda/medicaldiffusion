import copy
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import wandb
from pathlib import Path
from torch.optim import Adam

from torch.cuda.amp import autocast, GradScaler

from einops import rearrange

from ddpm.utils import  exists, cycle, noop, num_to_groups, video_tensor_to_gif
from ddpm.diffusion import EMA
from torch.utils.data import  DataLoader



class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset=None,
        val_dataset=None,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        validate_save_and_sample_every=1000,
        results_folder='./results',
        num_sample_rows=1,
        max_grad_norm=None,
        num_workers=20,
        conditioned=False,
        rank=0,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.validate_save_and_sample_every = validate_save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        
        self.ds = dataset
        self.val_dataset = val_dataset
        
        dl = DataLoader(self.ds, batch_size=train_batch_size,
                        shuffle=True, pin_memory=True, num_workers=num_workers)
        self.val_dl = DataLoader(self.val_dataset, batch_size=train_batch_size,
                                  shuffle=False, pin_memory=True, num_workers=num_workers)

        self.len_dataloader = len(dl)
        self.dl = cycle(dl)

        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.conditioned = conditioned

        self.rank = rank

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def training_step(self, prob_focus_present, focus_present_mask):
        # Training step
        for _ in range(self.gradient_accumulate_every):
            item = next(self.dl)
            data = item['data'].cuda()
            
            cond = item['cond'].cuda() if self.conditioned else None

            with autocast(enabled=self.amp):
                loss = self.model(
                    data,
                    prob_focus_present=prob_focus_present,
                    focus_present_mask=focus_present_mask,
                    cond=cond,
                )

                self.scaler.scale(loss / self.gradient_accumulate_every).backward()


        log = {'loss': loss.item()}

        if exists(self.max_grad_norm):
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)

        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()

        return log


    def validation_step(self, prob_focus_present, focus_present_mask):
        self.ema_model.eval()

        # Get milestone number
        milestone = self.step // self.validate_save_and_sample_every

        # Save milestone
        self.save(milestone)

        # Sample
        with torch.no_grad():
            # Get number of samples and their condition
            num_samples = self.num_sample_rows ** 2
            batches = num_to_groups(num_samples, self.batch_size)
            sample_cond = self.ds.get_cond(batch_size=batches[-1]).cuda() if self.conditioned else None
            print(f'Sampling {num_samples} images with condition {sample_cond}')

            # Sample the images
            all_videos_list = list(
                map(lambda n: self.ema_model.sample(batch_size=n, cond=sample_cond), batches))
            all_videos_list = torch.cat(all_videos_list, dim=0)

        # Save gif
        all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))
        one_gif = rearrange(
            all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=self.num_sample_rows)
        video_path = str(self.results_folder / str(f'{milestone}.gif'))
        video_tensor_to_gif(one_gif, video_path)

        # LOG gif to wandb
        log = {'gif': wandb.Video(video_path)}

        # Validate
        with torch.no_grad():
            for item in self.val_dl:
                data = item['data'].cuda()
                cond = item['cond'].cuda() if self.conditioned else None

                with autocast(enabled=self.amp):
                    val_loss = self.ema_model(
                        data,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask,
                        cond=cond,
                    )
        
        log = {**log, 'val_loss': val_loss.item()}

        return log


    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)

        if self.rank == 0:
            pbar = tqdm(total=self.train_num_steps)
        
        while self.step < self.train_num_steps:
            log = self.training_step(prob_focus_present, focus_present_mask)

            # Update EMA model when needed (only for rank 0)
            if self.rank == 0 and self.step % self.update_ema_every == 0:
                self.step_ema()

            # Validation step (only for rank 0)
            if self.rank == 0 and self.step != 0 and self.step % self.validate_save_and_sample_every == 0:
                val_log = self.validation_step(prob_focus_present, focus_present_mask)
                log = {**log, **val_log}

            log = {**log, 'global_step': self.step}

            # Log
            log_fn(log)
            
            self.step += 1

            if self.rank == 0 and self.step % 100 == 0:
                pbar.update(100)
        
        if self.rank == 0:
            pbar.close()

        print('training completed')
