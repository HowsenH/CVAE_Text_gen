import numpy as np
import utils as ut
from torch import nn, optim

def train(model, data_loader, tqdm, device, writer,
          iter_max=np.inf, iter_save=np.inf, reinitialize=False):
    # Optimization
    if reinitialize:
        try:
            model.reset_parameters()
        except AttributeError:
            pass
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    i = 0
    with tqdm(total=iter_max) as pbar:
        while True:
            for xu, yu in data_loader:
                i += 1 # i is num of gradient steps taken by end of loop iteration
                optimizer.zero_grad()

                xu = xu.to(device).reshape(xu.size(0), -1)
                yu = yu.to(device).reshape(xu.size(0), -1)
                G_inp = xu[:, 0:xu.size(1) - 1].clone()
                loss, summaries = model.loss(xu, yu, G_inp, i)

                loss.backward()
                optimizer.step()
                pbar.set_postfix(
                        loss='{:.3e}'.format(loss),
                        kl='{:.3e}'.format(summaries['gen/kl_z']),
                        rec='{:.3e}'.format(summaries['gen/rec']))
                pbar.update(1)

                # Log summaries
                if i % 50 == 0: ut.log_summaries(writer, summaries, i)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)

                if i == iter_max:
                    return

