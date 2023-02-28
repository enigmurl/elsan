import torch

for i in range(1980):
    u = []
    v = []
    for j in range(110):
        u.append(torch.load('ensemble/u_' + str(j) + '.pt')[i].clone())
        v.append(torch.load('ensemble/v_' + str(j) + '.pt')[i].clone())

    u = torch.stack(u)
    v = torch.stack(v)
    torch.save(u, 'ensemble/u_t' + str(i) + '.pt')
    torch.save(v, 'ensemble/v_t' + str(i) + '.pt')

    print("Finish", u.shape, v.shape, i)
