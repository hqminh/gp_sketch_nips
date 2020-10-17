from mvae import *
from gps.ssgp import *
from gps.fgp import *

class Experiment:
    def __init__(self, dataset='abalone', method='full', embedding=True, vae_model=None):
        self.dataset = dataset
        self.method = method
        self.embedding = embedding
        self.vae = vae_model

    @staticmethod
    def train_gp(gp_obj, test, n_iter=500, record_interval=10):
        idx = []
        nll = []
        error = []
        model = nn.ModuleList([gp_obj.cov, gp_obj.mean])
        optimizer = opt.Adam(model.parameters())
        for i in range(n_iter + 1):
            model.train()
            optimizer.zero_grad()
            loss = gp_obj.NLL()
            if i < n_iter + 1:
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            if record_interval == -1:
                continue

            elif (i + 1) % record_interval == 0:
                ypred, yvar = gp_obj(test['X'])
                error.append(rmse(ypred, test['Y']))
                nll.append(loss.item())
                idx.append(i + 1)
                print('Training Iteration', i + 1, 'rmse:', error[-1], 'nll:', nll[-1])

        if record_interval == -1:
            ypred, yvar = gp_obj(test['X'])
            return rmse(ypred, test['Y'])
        if record_interval == -2:
            return gp_obj.cov(gp_obj.data['X']), gp_obj.cov.weights
        else:
            return error, nll, idx

    @staticmethod
    def load_data(dataset):
        train = None
        test = None
        if dataset == 'abalone':
            train, n_train = abalone_data(is_train=True)
            test, n_test = abalone_data(is_train=False)

        elif dataset == 'gas500':
            full_train, full_test = gas_sensor_data(is_preload=True)
            p1 = torch.randperm(full_test['X'].size(0))
            idx_test = p1[:25000]
            test = {'X': full_test['X'][idx_test],
                    'Y': full_test['Y'][idx_test]
                    }
            p2 = torch.randperm(full_train['X'].size(0))
            idx_train = p2[:500000]
            train = {'X': full_train['X'][idx_train],
                     'Y': full_train['Y'][idx_train]
                     }
        elif dataset == 'gas10':
            full_train, full_test = gas_sensor_data(is_preload=True)
            p1 = torch.randperm(full_test['X'].size(0))
            idx_test = p1[:500]
            test = {'X': full_test['X'][idx_test],
                    'Y': full_test['Y'][idx_test]
                    }
            p2 = torch.randperm(full_train['X'].size(0))
            idx_train = p2[:10000]
            train = {'X': full_train['X'][idx_train],
                     'Y': full_train['Y'][idx_train]
                     }
        elif dataset == 'gas':
            train, test = gas_sensor_data(is_preload=True)
        return train, test

    @staticmethod
    def cluster_data(data, k=10):
        X = dt(data['X'])
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        cluster = {'k': k,
                   'label': kmeans.labels_,
                   'centroids': kmeans.cluster_centers_,
                   }
        for i in range(k):
            cluster[i] = {'idx': []}

        for i in range(X.shape[0]):
            cid = cluster['label'][i]
            cluster[cid]['idx'].append(i)

        for i in range(k):
            cluster[i]['X'] = data['X'][cluster[i]['idx']]
            cluster[i]['Y'] = data['Y'][cluster[i]['idx']]

        return cluster

    @staticmethod
    def create_gp_object(train, method):
        if ('ssgp' in method) or ('vaegp' in method):
            n_eps = int(method.split('_')[1])
            ssgp = SSGP(train['X'].shape[1], n_eps)
            return ssgp

        elif 'full' in method:
            fgp = FGP(train['X'].shape[1])
            return fgp