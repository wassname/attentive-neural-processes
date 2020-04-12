import neural_processes.utils
import pickle, json
import tempfile

def test_obectdict(tmpdir):
    o = neural_processes.utils.ObjectDict(z=1, b=4, test="g", w=0)
    pickle.dump(o, open(tmpdir+'/test.pkl', 'wb'))
    o2 = pickle.load(open(tmpdir+'/test.pkl', 'rb'))

    o3 = json.loads(json.dumps(o))
    print(o, o2, o3)
