import neural_processes.utils
import pickle, json
import torch
import tempfile

def test_obectdict(tmpdir):
    o = neural_processes.utils.ObjectDict(z=1, b=4, test="g", w=0)
    pickle.dump(o, open(tmpdir+'/test.pkl', 'wb'))
    o2 = pickle.load(open(tmpdir+'/test.pkl', 'rb'))

    o3 = json.loads(json.dumps(o))
    print(o, o2, o3)


def test_agg_logs():
    outputs = [
        {'val_loss': torch.tensor(0.7206),
            'log': {'val_loss': torch.tensor(0.7206), 'val_loss_p': torch.tensor(0.7206), 'val_loss_kl': torch.tensor(2.3812e-06), 'val_loss_mse': torch.tensor(0.1838)}},
        {'val_loss': torch.tensor(0.7047),
            'log': {'val_loss': torch.tensor(0.7047), 'val_loss_p': torch.tensor(0.7047), 'val_loss_kl': torch.tensor(2.8391e-06), 'val_loss_mse': torch.tensor(0.1696)}},
        ]
    r = neural_processes.utils.agg_logs(outputs)
    assert isinstance(r, dict)
    assert 'agg_val_loss' in r.keys()
    assert 'agg_val_loss_kl' in r['log'].keys()
    assert isinstance(r['agg_val_loss'], float)

    outputs = {'val_loss': torch.tensor(0.7206),
            'log': {'val_loss': torch.tensor(0.7206), 'val_loss_p': torch.tensor(0.7206), 'val_loss_kl': torch.tensor(2.3812e-06), 'val_loss_mse': torch.tensor(0.1838)}}
    r = neural_processes.utils.agg_logs(outputs)
    assert isinstance(r, dict)
    assert 'agg_val_loss' in r.keys()
    assert 'agg_val_loss_kl' in r['log'].keys()
    assert isinstance(r['agg_val_loss'], float)
