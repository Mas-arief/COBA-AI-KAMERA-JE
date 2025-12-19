import pickle, os
fn='simple_embeddings.pkl'
if not os.path.exists(fn):
    print('NO_FILE')
    raise SystemExit(0)
with open(fn,'rb') as f:
    m=pickle.load(f)
pe = m.get('person_embeddings',{})
print('persons_count=', len(pe))
for k,v in pe.items():
    print(k, 'len=', len(v), 'norm=', float((v@v)**0.5))
print('components shape=', m['components'].shape)
print('mean len=', len(m['mean']))
