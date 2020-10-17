from experiments import *
from torch.nn.utils.clip_grad import clip_grad_value_
import traceback
from pprint import pprint
import sys


class VAEGP(nn.Module):
    def __init__(self, train, test, vae_cluster=8, embed_dim=4, gp_method='vaegp_32', batch_size=200):
        super(VAEGP, self).__init__()
        self.device = get_cuda_device()
        self.dataset = TensorDataset(train['X'], train['Y'])
        self.testset = TensorDataset(test['X'], test['Y'])
        self.data = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.test = DataLoader(self.testset, batch_size=batch_size, shuffle=True)
        self.vae = MixtureVAE(self.data, train['X'].shape[1], embed_dim, vae_cluster, n_sample=5)
        self.original = train
        self.train = {'X': self.vae(train['X'], grad=False), 'Y': train['Y']}
        self.gp = Experiment.create_gp_object(self.original, gp_method)
        self.model = nn.ModuleList([self.vae, self.gp])
        self.history = []
        self.vae_params = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        self.gp_params = sum(p.numel() for p in self.gp.parameters() if p.requires_grad)
        print('No. of parameters: VAE = {} -- GP = {}'.format(self.vae_params, self.gp_params))

    def train_gp(self, seed=0, n_iter=100, n_epoch=50, lmbda=1.0, pred_interval=5, test=False, verbose=True):
        set_seed(seed)
        print('SEED=', seed)
        optimizer = opt.Adam(self.model.parameters())
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        pred_start = torch.cuda.Event(enable_timing=True)
        pred_end = torch.cuda.Event(enable_timing=True)
        pred_time = 0.0
        start.record()
        for i in range(n_iter):
            batch_nll = 0.0
            batch_elbo = 0.0
            batch_loss = 0.0
            epoch = 0
            for (X, Y) in self.data:
                if epoch > n_epoch:
                    break
                epoch += 1
                X.to(self.device)
                Y.to(self.device)
                Xa = self.vae(self.vae(X), encode=False)
                self.model.train()
                optimizer.zero_grad()
                delbo = self.vae.dsELBO(X, alpha=1.0, beta=1.0, gamma=1.0, verbose=False)
                nll = self.gp.NLL(Xa, Y)
                loss = - 0.01 * delbo + lmbda * nll
                batch_nll += nll * X.shape[0]
                batch_elbo += delbo * X.shape[0]
                batch_loss += loss * X.shape[0]
                loss.backward()
                clip_grad_value_(self.model.parameters(), 10)
                optimizer.step()
                torch.cuda.empty_cache()
            if i % pred_interval == 0:
                torch.cuda.synchronize()
                record = {'iter': i,
                          'nll': batch_nll.item() / self.train['X'].shape[0],
                          'elbo': batch_elbo.item() / self.train['X'].shape[0],
                          'loss': batch_loss.item() / self.train['X'].shape[0],
                          }
                if test:
                    pred_start.record()
                    Ypred_full = []
                    Yt_full = []
                    Xr = self.vae(self.vae(self.original['X'], grad=False), encode=False, grad=False)
                    for (Xt, Yt) in self.test:
                        Xtr = self.vae(self.vae(Xt, grad=False), encode=False, grad=False)
                        Ypred = self.gp(Xtr, Xr, self.original['Y'], var=False)
                        Ypred_full.append(Ypred)
                        Yt_full.append(Yt)
                        del Xtr
                        torch.cuda.empty_cache()
                    Ypred_full = torch.cat(Ypred_full)
                    Yt_full = torch.cat(Yt_full)
                    record['rmse'] = rmse(Ypred_full, Yt_full).item()
                    pred_end.record()
                    torch.cuda.synchronize()
                    pred_time += pred_start.elapsed_time(pred_end)
                    del Xr
                    torch.cuda.empty_cache()
                end.record()
                torch.cuda.synchronize()
                record['time'] = start.elapsed_time(end) - pred_time
                if verbose:
                    print(record)
                self.history.append(record)
        return self.history

    def forward(self, X):
        return self.gp(X, self.original['X'], self.original['Y'])

class GP_wrapper(nn.Module):
    def __init__(self, train, test, gp_method='ssgp_32', batch_size=200):
        super(GP_wrapper, self).__init__()
        self.device = get_cuda_device()
        self.train = train
        self.gp_method = gp_method
        self.gp = Experiment.create_gp_object(self.train, self.gp_method)
        self.dataset = TensorDataset(self.train['X'], self.train['Y'])
        self.testset = TensorDataset(test['X'], test['Y'])
        self.data = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.test = DataLoader(self.testset, batch_size=batch_size, shuffle=True)
        self.model = nn.ModuleList([self.gp.cov, self.gp.mean])
        self.history = []

    def train_gp(self, seed=0, n_iter=300, pred_interval=5, test=False, verbose=True, n_epoch=20):
        set_seed(seed)
        print('SEED=', seed)
        optimizer = opt.Adam(self.model.parameters())
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        pred_start = torch.cuda.Event(enable_timing=True)
        pred_end = torch.cuda.Event(enable_timing=True)
        pred_time = 0.0
        start.record()
        for i in range(n_iter):
            batch_nll = 0.0
            epoch = 0
            for (X, Y) in self.data:
                if epoch > n_epoch:
                    break
                epoch += 1
                self.model.train()
                optimizer.zero_grad()
                nll = self.gp.NLL(X, Y)
                batch_nll += nll * X.shape[0]
                nll.backward()
                clip_grad_value_(self.model.parameters(), 10)
                optimizer.step()
                torch.cuda.empty_cache()
            if i % pred_interval == 0:
                record = {'nll': batch_nll.item() / self.train['X'].shape[0],
                          'iter': i}
                if test:
                    pred_start.record()
                    Ypred_full = []
                    Yt_full = []
                    count = 0
                    for (Xt, Yt) in self.test:
                        count += 1
                        Yt_full.append(Yt)
                        Ypred = self.gp(Xt, self.train['X'], self.train['Y'], var=False)
                        Ypred_full.append(Ypred)

                    Ypred_full = torch.cat(Ypred_full)
                    Yt_full = torch.cat(Yt_full)
                    record['rmse'] = rmse(Ypred_full, Yt_full).item()
                    pred_end.record()
                    torch.cuda.synchronize()
                    pred_time += pred_start.elapsed_time(pred_end)
                end.record()
                torch.cuda.synchronize()
                record['time'] = start.elapsed_time(end) - pred_time
                if verbose:
                    print(record)
                self.history.append(record)
        return self.history


def deploy(num_seed, prefix, method, dataset, batch_size):
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    with open(prefix + method + '.res', 'w+') as f:
        train, test = Experiment.load_data(dataset)
        seed = [np.random.randint(10000) for _ in range(num_seed)]
        res = dict()
        for s in seed:
            if 'vaegp' in method:
                vaegp = VAEGP(train, test, gp_method=method, batch_size=batch_size)
                try:
                    res[s] = vaegp.train_gp(seed=s, n_iter=101, lmbda=1.0, pred_interval=10, test=True, verbose=True)
                    torch.save(vaegp, prefix + str(s) + '_' + method + '.pth')
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    res[s] = vaegp.history
                    torch.save(vaegp, prefix + str(s) + '_' + method + '.pth')
            else:
                try:
                    gp = GP_wrapper(train, test, gp_method=method, batch_size=batch_size)
                    res[s] = gp.train_gp(seed=s, n_iter=101, pred_interval=10, test=True, verbose=True)
                    torch.save(gp, prefix + str(s) + '_' + method + '.pth')
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    res[s] = gp.history
                    torch.save(gp, prefix + str(s) + '_' + method + '.pth')
            pprint(s, f)
            pprint(res[s], f)
            f.flush()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        torch.cuda.set_device(0)
        np.random.seed(1234)
        torch.manual_seed(1234)
        deploy(5, prefix='./results/gas_full/', method='vaegp_32', dataset='gas', batch_size=512)
    else:
        torch.cuda.set_device(int(sys.argv[4]))
        deploy(sys.argv[5], sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[6]))
