from utils.utility import *

import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

class GaussianNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=10, norm=True, bias=None):
        super(GaussianNet, self).__init__()
        self.device = get_cuda_device()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.fc21 = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.fc22 = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.bn = nn.BatchNorm1d(hidden_dim).to(self.device)
        self.norm = norm
        if bias is None:
            self.bias = torch.zeros(1, output_dim).to(self.device)
        else:
            self.bias = bias

    def forward(self, x):
        x = self.fc1(x)
        x_reshape = x.view(-1, self.fc1.out_features)
        if (x_reshape.shape[0] > 1) and (self.norm is True):
            shape = x.shape
            x = x.view(-1, self.fc1.out_features)
            x = F.relu(self.bn(x)).view(shape)
        else:
            x = F.relu(x)
        mu, log_var = self.fc21(x), F.tanh(self.fc22(x))  # mu: n_data by output_dim; log_var: n_data by output_dim
        return (mu + self.bias), log_var

    def sample(self, x, n_sample=10):  # previously, 100
        x = self.fc1(x)
        x_reshape = x.view(-1, self.fc1.out_features)
        if (x_reshape.shape[0] > 1) and (self.norm is True):
            shape = x.shape
            x = x.view(-1, self.fc1.out_features)
            x = F.relu(self.bn(x)).view(shape)
        else:
            x = F.relu(x)
        mu, log_var = self.fc21(x), F.tanh(self.fc22(x))  # mu: n_data by output_dim; log_var: n_data by output_dim
        std = torch.exp(0.5 * log_var)
        eps = torch.empty(n_sample, mu.shape[0], self.output_dim).normal_(0, 1).to(self.device)
        return (mu + self.bias) + eps * std


class MixtureGaussianNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_component, prior_bias=False):
        super(MixtureGaussianNet, self).__init__()
        self.device = get_cuda_device()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_component = n_component
        self.components = nn.ModuleList()
        for i in range(self.n_component):
            bias = None
            if prior_bias:
                bias = torch.empty(1, self.output_dim).uniform_(-1, 1).to(self.device)
                bias = 10.0 * bias / torch.norm(bias)
            self.components.append(GaussianNet(input_dim, output_dim, bias=bias))
        self.weights = nn.Linear(input_dim, n_component).to(self.device)

    def forward(self, Z, X=None, grad=True):  # return log p(z|x) for each pair (z, x) or log p(z) for each z if X is None
        def _forward(X):
            if X is None:
                X = torch.zeros((Z.shape[0], self.input_dim)).to(self.device)
            res = torch.zeros((Z.shape[0], self.n_component)).to(self.device)
            for i in range(self.n_component):
                mean, logvar = self.components[i](X)  # mean, logvar: Z.shape[0] by Z.shape[1]
                res[:, i] -= 0.5 * torch.sum(((mean - Z) ** 2) * torch.exp(-2.0 * logvar), dim=1)
                res[:, i] -= 0.5 * torch.sum(logvar, dim=1)

            w = F.softmax(self.weights(torch.ones(1, self.input_dim).to(self.device)))
            res += torch.log(w)
            return torch.logsumexp(res, dim=1, keepdim=True)  # output has shape Z.shape[0] by 1

        if not grad:
            with torch.no_grad():
                return _forward(X)
        else:
            return _forward(X)

    def sample(self, X=None, n_sample=10):  # return n_sample from p(.|x) or p(.) if X is None
        if X is None:
            X = torch.zeros((1, self.input_dim)).to(self.device)

        U = torch.empty(n_sample, self.n_component).uniform_(0, 1).to(self.device)
        w = F.softmax(self.weights(torch.ones(1, self.input_dim).to(self.device)))
        chosen = torch.argmax(torch.log(w[:, :] / (1.0 - w[:, :])) - torch.log(-torch.log(U)), dim=1)
        c_sample = torch.cat([c.sample(X, n_sample=1) for c in self.components], dim=0)
        res = c_sample[chosen]
        return res


class MixtureVAE(nn.Module):
    def __init__(self, dataset, input_dim, output_dim, n_component, n_sample=10):  # previously, 100
        super(MixtureVAE, self).__init__()
        self.device = get_cuda_device()
        self.dataset = dataset
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_component = n_component
        self.n_sample = n_sample
        self.qz_x = MixtureGaussianNet(self.input_dim, self.output_dim, self.n_component, prior_bias=True)
        self.px_z = GaussianNet(self.output_dim, self.input_dim, bias=None)
        self.pz = MixtureGaussianNet(self.input_dim, self.output_dim, self.n_component, prior_bias=True)

    def dsELBO(self, X, alpha=8.0, beta=1.2, gamma=1.0, verbose=False):
        Z = self.qz_x.sample(X, n_sample=self.n_sample)  # n_sample by X.shape[0] by output_dim
        Z = Z.reshape(self.n_sample * X.shape[0], -1)
        Xr = X.repeat(self.n_sample, 1)

        qzx = self.qz_x(Z, Xr)
        log_qzx = torch.sum(qzx) / Xr.shape[0]
        log_qz = torch.logsumexp(qzx, dim=0)[0] - np.log(Xr.shape[0])
        log_pz = torch.sum(self.pz(Z)) / Xr.shape[0]
        mu_pxz, var_pxz = self.px_z(Z)
        log_pxz = (-0.5 * (torch.sum((Xr - mu_pxz) ** 2 * torch.exp(-2.0 * var_pxz) + var_pxz))) / Xr.shape[0]

        qizx_mu = []
        qizx_log_var = []
        for j in range(self.n_component):
            mu, var = self.qz_x.components[j](X)
            qizx_mu.append(mu.view(1, mu.shape[0], mu.shape[1]))
            qizx_log_var.append(var.view(1, var.shape[0], var.shape[1]))

        qczx_mu = torch.mean(torch.cat(qizx_mu), dim=0)
        qczx_log_var = torch.mean(torch.cat(qizx_log_var), dim=0)
        kl_separation = torch.zeros(self.n_component).to(self.device)
        for j in range(self.n_component):
            ld = torch.sum(qizx_log_var[j]) - torch.sum(qczx_log_var)
            tr = torch.sum(torch.exp(qczx_log_var - qizx_log_var[j]))
            qd = torch.sum((qczx_mu - qizx_mu[j]) ** 2 / torch.exp(-1.0 * qczx_log_var))
            kl_separation[j] = ld + tr + qd

        # ELBO components
        elbo = log_pxz
        elbo_alpha = log_qz - log_pz
        elbo_beta = log_qzx - log_pz
        elbo_gamma = 0.5 * torch.min(kl_separation) / X.shape[0]
        if verbose: print(elbo.item(), elbo_alpha.item(), elbo_beta.item(), elbo_gamma.item())
        return elbo - beta * elbo_beta - alpha * elbo_alpha + gamma * elbo_gamma

    def forward(self, data, encode=True, grad=True):
        def _forward():
            if encode is True:
                res = self.qz_x.sample(data)
                return torch.mean(res, dim=0)  # data.shape[0] by output_dim
            else:
                res, _ = self.px_z(data)  # data.shape[0] by input_dim
                return res

        if not grad:
            with torch.no_grad():
                return _forward()
        else:
            return _forward()

    def train_dsvae(self, n_iter=100, alpha=8.0, beta=1.2, gamma=1.0, verbose=True):
        optimizer = opt.Adam(self.model.parameters())
        for i in range(n_iter):
            ave_loss = 0.0
            for (X, _) in self.dataset:
                self.model.train()
                optimizer.zero_grad()
                delbo = self.dsELBO(X.to(self.device), alpha, beta, gamma, verbose=verbose)
                loss = -delbo
                ave_loss += loss / X.shape[0]
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            print('Ave Loss=', ave_loss)
