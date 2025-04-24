import torch
import torch.nn as nn

def gram_linear(features):
    return features @ features.T

def center_gram(gram):
    n = gram.shape[0]
    identity = torch.eye(n, device=gram.device)
    ones = torch.ones((n, n), device=gram.device) / n
    Hm = identity - ones
    return Hm @ gram @ Hm

def compute_cka(gram_x, gram_y):
    numerator = torch.trace(gram_x @ gram_y)
    denominator = torch.sqrt(torch.trace(gram_x @ gram_x) * torch.trace(gram_y @ gram_y))
    return numerator / denominator

def extract_features(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        features = model(data)
    return features

def compute_cka_between_minima(model1, model2, data, device='cpu'):
    model1.to(device)
    model2.to(device)

    features1 = extract_features(model1, data, device)
    features2 = extract_features(model2, data, device)

    gram_x = gram_linear(features1)
    gram_y = gram_linear(features2)

    gram_x = center_gram(gram_x)
    gram_y = center_gram(gram_y)

    cka_score = compute_cka(gram_x, gram_y)
    
    return cka_score.item()