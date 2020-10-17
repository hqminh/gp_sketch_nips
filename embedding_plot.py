from utils.utility import *
from experiments import *
from vaegp import *
from sklearn.manifold import TSNE


def cluster_coloring(X, savename, k=8):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    clusters_x = []
    clusters_y = []
    for i in range(k):
        clusters_x.append([])
        clusters_y.append([])
    for i in range(X.shape[0]):
        clusters_x[kmeans.labels_[i]].append(X[i][0])
        clusters_y[kmeans.labels_[i]].append(X[i][1])

    color = ['red', 'blue', 'green', 'navy', 'turquoise', 'darkorange', 'black', 'purple']

    plt.figure()
    for i in range(k):
        plt.scatter(clusters_x[i], clusters_y[i], color=color[i])
    plt.savefig(savename)


def main():
    prefix = './results_to_be_processed/gas10/'
    seed = ['108', '263', '411', '807', '2010']
    samp = ['16', '64', '128']
    for s in seed:
        for p in samp:
            try:
                vaegp = torch.load(prefix + s + '_vaegp_' + p + '.pth', map_location=get_cuda_device())
                train, test = Experiment.load_data('gas10')
                Xr = dt(vaegp.vae(vaegp.vae(train['X'], grad=False), encode=False, grad=False))
                X = dt(train['X'])
                Xre = TSNE(n_components=2).fit_transform(Xr)
                Xe = TSNE(n_components=2).fit_transform(X)
                cluster_coloring(Xe, './cluster_visualization/' + s + '_' + p + 'embed_visual_original.png')
                cluster_coloring(Xre, './cluster_visualization/' + s + '_' + p + 'embed_visual_reconstruct.png')
            except Exception as e:
                print(e, s, p)

if __name__ == '__main__':
    main()