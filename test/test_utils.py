import neural_processes.utils
import pickle, json
import torch
import tempfile


def test_obectdict(tmpdir):
    o = neural_processes.utils.ObjectDict(z=1, b=4, test="g", w=0)
    pickle.dump(o, open(tmpdir + "/test.pkl", "wb"))
    o2 = pickle.load(open(tmpdir + "/test.pkl", "rb"))

    o3 = json.loads(json.dumps(o))
    print(o, o2, o3)


def test_agg_logs():
    outputs = [
        {
            "val_loss": torch.tensor(0.7206),
            "log": {
                "val_loss": torch.tensor(0.7206),
                "val_loss_p": torch.tensor(0.7206),
                "val_loss_kl": torch.tensor(2.3812e-06),
                "val_loss_mse": torch.tensor(0.1838),
            },
        },
        {
            "val_loss": torch.tensor(0.7047),
            "log": {
                "val_loss": torch.tensor(0.7047),
                "val_loss_p": torch.tensor(0.7047),
                "val_loss_kl": torch.tensor(2.8391e-06),
                "val_loss_mse": torch.tensor(0.1696),
            },
        },
    ]
    r = neural_processes.utils.agg_logs(outputs)
    assert isinstance(r, dict)
    assert "agg_val_loss" in r.keys()
    assert "agg_val_loss_kl" in r["log"].keys()
    assert isinstance(r["agg_val_loss"], float)

    outputs = {
        "val_loss": torch.tensor(0.7206),
        "log": {
            "val_loss": torch.tensor(0.7206),
            "val_loss_p": torch.tensor(0.7206),
            "val_loss_kl": torch.tensor(2.3812e-06),
            "val_loss_mse": torch.tensor(0.1838),
        },
    }
    r = neural_processes.utils.agg_logs(outputs)
    assert isinstance(r, dict)
    assert "agg_val_loss" in r.keys()
    assert "agg_val_loss_kl" in r["log"].keys()
    assert isinstance(r["agg_val_loss"], float)


def test_round_values():
    r = neural_processes.utils.round_values(
        {"a": 0.00004, "d": {"b": 124455.45, "c": 0.004}, "l": 500}
    )


def test_hparams_power():
    r = neural_processes.utils.hparams_power({"test_power": 2, "test2": 2})
    assert r["test"] == 2 ** 2
    assert r["test2"] == 2


def test_log_prob_sigma():
    mean = torch.zeros(4, 5)
    log_scale = torch.ones(4, 5)
    value = torch.zeros(4, 5)
    y_dist = torch.distributions.Normal(mean, log_scale.exp())
    r1 = y_dist.log_prob(value)
    r2 = neural_processes.utils.log_prob_sigma(value, mean, log_scale)
    assert (r1 == r2).all()


def test_kl_loss_var():
    prior_mu = torch.zeros(4, 5)
    post_mu = torch.zeros(4, 5) + 1
    log_var_prior = torch.ones(4, 5)
    log_var_post = torch.ones(4, 5) + 1
    dist_prior = torch.distributions.Normal(prior_mu, torch.exp(0.5 * log_var_prior))
    dist_post = torch.distributions.Normal(post_mu, torch.exp(0.5 * log_var_post))
    r1 = torch.distributions.kl_divergence(dist_post, dist_prior)
    r2 = neural_processes.utils.kl_loss_var(
        prior_mu, log_var_prior, post_mu, log_var_post
    )
    assert (r1 == r2).all()
