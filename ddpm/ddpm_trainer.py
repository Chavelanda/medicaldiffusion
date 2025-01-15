import copy
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import wandb
from pathlib import Path
from torch.optim import Adam

from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

from einops import rearrange

from ddpm.utils import  exists, cycle, noop, num_to_groups, video_tensor_to_gif
from ddpm.diffusion import EMA
from torch.utils.data import  DataLoader



class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dl=None,
        val_dl=None,
        ema_decay=0.995,
        train_batch_size=32,
        base_lr=1e-4,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every_n_steps=1000,
        check_val_every_n_epoch=1,
        results_folder='./results',
        num_sample_rows=1,
        max_grad_norm=None,
        num_workers=20,
        conditioned=False,
        null_cond_prob=0.,
        rank=0,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every_n_steps = save_and_sample_every_n_steps
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        
        self.ds = dl.dataset
        
        self.len_dataloader = len(dl)
        self.dl = cycle(dl)
        self.val_dl = val_dl

        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        start_factor = base_lr/train_lr
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.opt, start_factor=start_factor, total_iters=5)
        
        self.step = 0
        self.epoch = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.conditioned = conditioned
        self.null_cond_prob = null_cond_prob

        self.rank = rank

        self.best_val_loss = float('inf')

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
            'scaler': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
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
        self.best_val_loss = data['best_val_loss']

    def training_step(self, prob_focus_present, focus_present_mask):
        # Training step
        loss_acc = 0
        counter = 0
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
                    null_cond_prob=self.null_cond_prob,
                )

                self.scaler.scale(loss / self.gradient_accumulate_every).backward()

            loss_acc += loss.item()*data.shape[0]
            counter += data.shape[0]

        if exists(self.max_grad_norm):
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)

        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()

        return loss_acc, counter 


    def validation_step(self, prob_focus_present, focus_present_mask):
        self.ema_model.eval()

        with torch.no_grad():
            val_loss_acc = 0
            for item in self.val_dl:
                data = item['data'].cuda()
                cond = item['cond'].cuda() if self.conditioned else None

                with autocast(enabled=self.amp):
                    val_loss = self.ema_model.module(
                        data,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask,
                        cond=cond,
                        null_cond_prob=self.null_cond_prob,
                    )
                    val_loss_acc += val_loss.item() * data.shape[0]
            
            val_loss = val_loss_acc / len(self.val_dl.dataset)
        # Save if best model
        if val_loss < self.best_val_loss:
            print('New best model!')
            self.best_val_loss = val_loss
            self.save('best')

        log = {'val_loss': val_loss}

        return log
    
    
    def save_and_sample(self):
        self.ema_model.eval()
        
        # Save milestone
        milestone = self.step // self.save_and_sample_every_n_steps
        self.save(milestone)

        # Sample
        with torch.no_grad():
            # Get number of samples and their condition
            num_samples = self.num_sample_rows ** 2
            batches = num_to_groups(num_samples, self.batch_size)
            sample_cond = self.ds.get_cond(batch_size=batches[-1]).cuda() if self.conditioned else None
            print(f'Sampling {num_samples} images with condition {sample_cond}')

            # Sample the images
            all_videos_list = list(map(lambda n: self.ema_model.module.sample(batch_size=n, cond=sample_cond), batches))
            all_videos_list = torch.cat(all_videos_list, dim=0)

        # Save gif
        all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))
        one_gif = rearrange(
            all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=self.num_sample_rows)
        video_path = str(self.results_folder / str(f'{milestone}.gif'))
        video_tensor_to_gif(one_gif, video_path)

        # LOG gif to wandb
        log = {'gif': wandb.Video(video_path)}

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
            pbar.update(self.step)
        
        loss_acc = 0
        counter = 0        
        while self.step < self.train_num_steps:
            validate_or_sampled = False

            loss_step, c_step = self.training_step(prob_focus_present, focus_present_mask)
            loss_acc += loss_step
            counter += c_step
            log = {'loss': loss_acc / counter}

            # Update EMA model when needed (only for rank 0)
            if self.rank == 0 and self.step % self.update_ema_every == 0:
                self.step_ema()

            # Validation step (only for rank 0). Does not happen at beginning. Happens at end of training epoch every check_val_every_n_epoch number of epochs
            if self.rank == 0 and self.step != 0 and self.step % self.len_dataloader == 0 and self.epoch % self.check_val_every_n_epoch == 0:
                print('\nValidating!')
                validate_or_sampled = True
                val_log = self.validation_step(prob_focus_present, focus_present_mask)
                log = {**log, **val_log}
                print('\nValidation over!')
            dist.barrier()

            # Save and sample (only for rank 0). Does not happen at beginning. Happens every save_and_sample_every_n_steps steps
            if self.rank == 0 and self.step != 0 and self.step % self.save_and_sample_every_n_steps == 0:
                print('\nSampling!')
                validate_or_sampled = True
                sample_log = self.save_and_sample()
                log = {**log, **sample_log}
            dist.barrier()

            log = {**log, 'global_step': self.step, 'learning_rate': self.scheduler.get_last_lr()[0]}

            # Log every 50 steps or when validation or sampling occurred
            if self.step % 50 == 0 or validate_or_sampled:
                log_fn(log)
                loss_acc, counter = 0, 0

            # Update scheduler if epoch is over. Update epoch counter
            if self.step % self.len_dataloader == 0 and self.step != 0:
                self.scheduler.step()
                self.epoch += 1

            self.step += 1
            validate_or_sampled = False

            if self.rank == 0 and self.step % 50 == 0:
                pbar.update(50)
        
        if self.rank == 0:
            pbar.close()

        print('training completed')
