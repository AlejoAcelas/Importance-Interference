import pytest
from model_settings import *
import torch
from jaxtyping import Float, Int, Bool

### Batch generators ###

@pytest.fixture
def bare_model():
    config = Config(n_features=5, n_hidden=2, n_instances=3)
    model = BasicMLP(config, device='cpu')
    return model

@pytest.fixture
def model_with_params():
    config = Config(n_features=5, n_hidden=2, n_instances=3)
    feature_probability = torch.tensor([0.0, 0.5, 1.0])
    importance = torch.tensor([1, 0.5, 0, 0, 0])
    model = BasicMLP(config, feature_probability[:, None], importance[None, :], device='cpu')
    return model

def check_batch_freq(batch):
    # Check that it handles features probabilities correctly
    # It's harcoded for now
    batch = batch > 0
    batch_freq = batch.float().mean(dim=(0, 2))
    assert batch_freq[0] == 0.0
    assert batch_freq[1] > 0.4 and batch_freq[1] < 0.6
    assert batch_freq[2] == 1.0

def test_generate_batch_rand(model_with_params):
    model = model_with_params
    batch, target = model.generate_batch_rand(1000)
    
    assert batch.shape == (1000, 3, 5)
    torch.testing.assert_close(target, batch)    
    check_batch_freq(batch)

def test_generate_batch_one_hot_noiseless(bare_model, model_with_params):
    model = model_with_params
    bare_batch, bare_target = bare_model.generate_batch_one_hot_noiseless(1000)
    batch, target = model.generate_batch_one_hot_noiseless(1000)

    assert (bare_batch == batch).all()
    assert (bare_target == target).all()

    assert batch.shape == (1000, 3, 5)
    assert target.shape == (1000, 3)
    assert (batch.argmax(-1) == target).all()
    assert (batch[:5, 0, :] == torch.eye(5)).all()


def test_generate_batch_one_hot(model_with_params):
    model = model_with_params
    batch, target = model.generate_batch_one_hot(1000)

    assert batch.shape == (1000, 3, 5)
    assert target.shape == (1000, 3)
    assert batch.gather(-1, target[..., None]).all() # Check that target always correspond to an active feature
    check_batch_freq(batch[:, :, :-1]) # Exclude the last feature, which behaves differently

### Loss functions ###

def test_mse_loss(model_with_params, bare_model):
    model = model_with_params
    batch, target = model.generate_batch_rand(1000)
    batch_one_hot, _ = model.generate_batch_one_hot_noiseless(5)

    zero_loss = model.mse_loss(batch, target)
    loss_per_feat_inst = model.mse_loss(batch_one_hot, torch.zeros_like(batch_one_hot), per_feature=True).mean(0) # [n_instances, n_features]
    loss_per_feat = loss_per_feat_inst.mean(dim=0)

    assert zero_loss.shape == (3,)
    assert loss_per_feat_inst.shape == (3, 5)
    torch.testing.assert_close(zero_loss, torch.zeros(3,))
    torch.testing.assert_close(loss_per_feat[0]/loss_per_feat[1], torch.tensor(2.0), rtol=0.1, atol=0.1)
    torch.testing.assert_close(loss_per_feat[2:], torch.zeros(3), rtol=0.1, atol=0.1)

    loss_per_feat_one_weighted = bare_model.mse_loss(batch, torch.zeros_like(batch), per_feature=True)
    loss_per_feat_unweighted = model.mse_loss_unweighted(batch, torch.zeros_like(batch), per_feature=True)

    torch.testing.assert_close(loss_per_feat_one_weighted, loss_per_feat_unweighted)

def test_cross_entropy_loss(model_with_params, bare_model):
    model = model_with_params
    batch_one_hot, target_one_hot = model.generate_batch_one_hot_noiseless(5)
    batch, target = model.generate_batch_one_hot(1000)
    
    loss = model.cross_entropy_loss_unweighted(batch, target, per_feature=False)
    loss_per_feat = bare_model.cross_entropy_loss(batch, target, per_feature=True).mean(0)
    loss_per_feat_unweighted = model.cross_entropy_loss_unweighted(batch, target, per_feature=True).mean(0)
    low_loss_per_feat = model.cross_entropy_loss(20*batch_one_hot, target_one_hot, per_feature=True).mean(0)

    print('Loss', loss) 
    print('Loss per feat', loss_per_feat)
    print('Loss per feat sum/mean', loss_per_feat.sum(-1), loss_per_feat.mean(-1))

    assert low_loss_per_feat.shape == (3, 5)
    torch.testing.assert_close(low_loss_per_feat, torch.zeros(3, 5))
    torch.testing.assert_close(loss_per_feat, loss_per_feat_unweighted)
    torch.testing.assert_close(loss, loss_per_feat.sum(-1))    

def test_batch_together(model_with_params):
    batch, target = batch_together(model_with_params.generate_batch_rand, 10, n_batch=100)
    batch_one_hot, target_one_hot = batch_together(model_with_params.generate_batch_one_hot, 10, n_batch=100)

    assert batch.shape == (100, 10, 3, 5)
    assert target.shape == (100, 10, 3, 5)
    assert batch_one_hot.shape == (100, 10, 3, 5)
    assert target_one_hot.shape == (100, 10, 3)

