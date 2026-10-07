"""Finite prefix classifiers; all trainable quantities and probabilities are real."""
import hashlib
import numpy as np


def reverse4(value):
    return int(f"{int(value):04b}"[::-1], 2)


def taxonomy(domains):
    names = sorted(domains)
    labels, group, codes, withheld = [], [], [], []
    for d, name in enumerate(names):
        intents = sorted(domains[name])
        hold = min(intents, key=lambda x: hashlib.sha256(
            ("structural-intent-heldout-v1|"+name+"|"+x).encode()).hexdigest())
        withheld.append(hold)
        for j, intent in enumerate(intents):
            if intent != hold:
                labels.append(intent); group.append(d)
                codes.append(reverse4(d)+16*reverse4(j))
    return names, labels, np.array(group), np.array(codes), withheld


def paths(codes):
    """Omit unary branches: C occupied leaves have exactly C-1 free heads."""
    codes = np.asarray(codes, dtype=np.int64)
    nodes, left, right = [], [], []
    for k in range(8):
        for prefix in sorted(set((codes % (2**k)).tolist())):
            member = codes % (2**k) == prefix
            bit = (codes >> k) & 1
            if np.any(member & (bit == 0)) and np.any(member & (bit == 1)):
                nodes.append((k, prefix))
                left.append(member & (bit == 0)); right.append(member & (bit == 1))
    assert len(nodes) == len(codes)-1
    return np.array(nodes), np.array(left, dtype=np.float64), np.array(right, dtype=np.float64)


def softmax(z):
    z = z-z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e/e.sum(axis=1, keepdims=True)


def probabilities(x, w, b, kind, left=None, right=None):
    z = np.asarray(x, dtype=np.float64)@w.T+b
    if kind == "tree":
        logp = -np.logaddexp(0, z)@left-np.logaddexp(0, -z)@right
        return np.exp(logp)
    return softmax(np.column_stack((z, np.zeros(len(z)))))


def objective_gradient(x, y, group, w, b, kind, left=None, right=None):
    """Analytic real gradient, also used for numerical checks before fitting."""
    z = x@w.T+b
    p = probabilities(x, w, b, kind, left, right)
    d = group[y]
    if kind == "domain":
        loss = -np.log(p[np.arange(len(y)), d]).mean()
        grad = p.copy(); grad[np.arange(len(y)), d] -= 1
        grad = grad[:, :-1]
    else:
        membership = np.eye(10)[group]
        q = p@membership
        loss = (-np.log(p[np.arange(len(y)), y])-np.log(q[np.arange(len(y)), d])).mean()
        conditional = p*(group[None, :] == d[:, None])/q[np.arange(len(y)), d, None]
        if kind == "tree":
            sig = 1/(1+np.exp(-z))
            mask = left+right
            grad = sig*mask[:, y].T-right[:, y].T
            grad += sig*(conditional@mask.T)-conditional@right.T
        else:
            grad = 2*p-conditional
            grad[np.arange(len(y)), y] -= 1
            grad = grad[:, :-1]
    loss += .0005*np.sum(w*w)
    return float(loss), grad.T@x/len(y)+.001*w, grad.mean(axis=0)


def metrics(p, y, d, group, kind):
    q = p if kind == "domain" else p@np.eye(10)[group]
    result = dict(domain_accuracy=float(np.mean(q.argmax(axis=1) == d)),
                  domain_nll=float(-np.log(q[np.arange(len(d)), d]).mean()),
                  probability_sum_max_error=float(np.max(np.abs(p.sum(axis=1)-1))))
    if kind != "domain" and np.all(y >= 0):
        result.update(intent_accuracy=float(np.mean(p.argmax(axis=1) == y)),
                      intent_nll=float(-np.log(p[np.arange(len(y)), y]).mean()))
    return result
