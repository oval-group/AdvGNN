import torch
import torch.nn as nn
import torch.distributions as dist
from adv_exp.attack_class import Attack_Class


#######################################################################################################
#   Implementation of the standard PGD attack
#   TODO:
#       add a few more comments
#######################################################################################################


default_params = {
        'iters': 40,
        'optimizer': 'default',
        'num_adv_ex': 5,
        'lr': 1e-4,
        'check_adv': 100,
    }


class Pgd_Attack(Attack_Class):

    def __init__(self, params=None, cpu=False, store_loss_progress=False):
        self.__name__ = 'PGD_attack'
        self.params = dict(default_params, **params) if params is not None else default_params
        self.cpu = cpu
        self.store_loss_progress = store_loss_progress

    def create_adv_examples(self, data, model, return_criterion="all", init_tensor=None,
                            target=None, gpu=False, return_iters=False):
        with torch.enable_grad():
            assert return_criterion in ["one", "half", "all", "not_early"]
            # self.targeted_attack = type(target) != type(None)
            self.targeted_attack = not isinstance(target, type(None))

            x, y, x_lbs, x_ubs = data
            if gpu and torch.cuda.is_available():
                x = x.cuda()
                x_lbs = x_lbs.cuda()
                x_ubs = x_ubs.cuda()
                model.cuda()
            device = x.device

            iters = self.params['iters']
            num_adv = self.params['num_adv_ex']
            alpha = 1e-5

            if device.type == 'cpu':
                labels = torch.LongTensor([y]*num_adv, device=device)
            else:
                labels = torch.cuda.LongTensor([y]*num_adv, device=device)

            # Calculate the mean of the normal distribution in logit space
            prior = dist.Uniform(low=x_lbs, high=x_ubs)
            images = prior.sample(torch.Size([num_adv]))   # Alg1 line 2

            if not isinstance(init_tensor, type(None)):
                if images[0].size() == init_tensor.size():
                    images[0] = init_tensor
                    # print("only initialized the initial tensor")
                elif images[0].size() == init_tensor[0].size():
                    # print("before", images.size(), "after", init_tensor.size())
                    images = init_tensor
                    # print("initialized the whole tensor with initial tensor")
                else:
                    print("image size", images.size(), images[0].size())
                    print("init tensor size", init_tensor.size(), init_tensor[0].size())
                    input("images and init tensor not compatible")

            if self.params['optimizer']:
                if self.params['optimizer'] == 'default':
                    alpha = self.params['lr']
                    images.requires_grad = True
                elif self.params['optimizer'] == 'adam':
                    images.requires_grad = True
                    model.zero_grad()
                    self.optimizer = torch.optim.Adam([images], lr=self.params['lr'], weight_decay=False)
                elif self.params['optimizer'] == 'SGLD':
                    self.optimizer = SGLD([images], lr=self.params['lr'], weight_decay=False, noise_scale=1e-4)
                    lr_ = self.params['lr']
                    a = 100
                    b = 1
                    gamma = 0.8
                else:
                    print("optimizer", self.params['optimizer'])
                    raise NotImplementedError

            if not isinstance(target, type(None)):
                self.loss_type = 'targeted_loss'
            else:
                self.loss_type = 'CE_loss'
                self.CE_loss = nn.CrossEntropyLoss()
            loss = nn.CrossEntropyLoss()

            self.loss_progress = []

            for i in range(iters):
                if self.params['optimizer'] == 'default':
                    images.requires_grad = True
                    outputs = model(images)

                    model.zero_grad()
                    cost = self._loss(outputs, labels, target).to(device)
                    cost.backward()

                    adv_images = images + alpha*images.grad.sign()
                    images = torch.max(torch.min(adv_images, x_ubs), x_lbs).detach_()

                elif self.params['optimizer'] == 'adam':
                    images.requires_grad = True
                    self.optimizer.zero_grad()

                    outputs = model(images)

                    cost = self._loss(outputs, labels, target).to(device)
                    cost.backward(retain_graph=True)

                    self.optimizer.step()
                    images = torch.max(torch.min(images, x_ubs), x_lbs).detach_()

                elif self.params['optimizer'] == 'SGLD':
                    if True:
                        lr_ = a*((b+i)**(-gamma))
                        self.adjust_lr(self.optimizer, lr_)
                        self.adjust_noise(self.optimizer, lr_**2)
                    images.requires_grad = True
                    images_clamp = torch.max(torch.min(images, x_ubs), x_lbs)
                    self.optimizer.zero_grad()

                    outputs = model(images_clamp)
                    cost = - loss(outputs, labels).to(device)
                    cost.backward()

                    self.optimizer.step()

                if self.store_loss_progress:
                    self.loss_progress.append(cost.detach())

                if i % self.params['check_adv'] == 0:
                    outputs = model(images)
                    succ, sum_, mean_ = self.success_tensor(outputs, y, target)
                    if return_criterion == "all" and mean_ == 1:
                        break
                    elif return_criterion == "one" and mean_ > 0:
                        print("return early, iter ", i)
                        break
                    elif return_criterion == "half" and mean_ >= 0.5:
                        break

            succ, sum_, mean_ = self.success_tensor(outputs, y, target)

            if return_iters:
                return images, succ, i
            else:
                return images, succ
