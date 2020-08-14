# bilinear attention model and residual learning
# reference: https://papers.nips.cc/paper/7429-bilinear-attention-networks.pdf
import numpy as np
import copy
def BAN(X, Y, U, V, P):
    """
    bilinear attention model forward pesudo-process
    ==========
    Parameters:
    ------------
    Input Variables:
    X : array (n_X_feats, n_X_channels)
    Y : array (n_Y_feats, n_Y_channels)

    Learnable Variables:
    U : array (n_X_feats, n_att_dim)
    V : array (n_Y_feats, n_att_dim)
    P : array (n_att_dim, n_glimpses)

    ===========
    Returns
    ------------
    f : array (n_att_dim, )
    """
    F = copy.copy(X.T) # if K != N
    for i in range(P.shape[1]):
        A = F.T @ U[i] * P[:,i] @ V.T @ Y
        A = np.exp(A) / np.sum(np.exp(A)) # bilinear pool
        f = np.diag(U.T @ F @ A @ Y.T @ V) # bilinear attention
        F = f + F
    f = np.sum(F, axis=-1)
    print(f.shape)
    return f

if __name__ == "__main__":
    X = np.zeros((100, 10))
    Y = np.zeros((200, 6))
    U = np.zeros((100, 50))
    V = np.zeros((200, 50))
    P = np.zeros((50, 8))
    BAN(X, Y, U, V, P)
